#!/usr/bin/env python3
# eval_llm_metrics.py
#
# Evaluate Groq LLaMA JSONL outputs for:
#  - 4.4.1 JSON Structural Validity Score (JSVS)
#  - 4.4.2 Schema Match Rate (SMR) + field completeness before/after enrichment
#  - 4.4.3 Hallucination Detection Metric (HDM, heuristic via low keyword overlap)
#  - 4.4.4 Latency Score (LS)   [uses per-record latency_ms if present, otherwise None]
#  - 4.4.5 Throughput Score     [records per minute, approximated from processed_at]
#  - 4.4.6 Factual Consistency Evaluation (FCE, average Jaccard keyword overlap)
#
# This script ONLY evaluates the CURRENT Groq LLaMA model used in
# llm_enrich_drugbank.py. It expects JSONL outputs written by that script in
# MinIO / S3 bucket 'bronze' under prefix 'gold/drugbank_llm/'.
#
# Usage (from project_demo/):
#   python scripts/eval_llm_metrics.py
#
# Requirements:
#   pip install boto3 pydantic
#
# Notes:
#   - HDM here is a heuristic: low lexical overlap is treated as
#     "potential hallucination". True hallucination detection still
#     requires domain expert review.
#   - LS is accurate only if you instrument llm_enrich_drugbank.py to
#     record per-call latency in a 'metrics.latency_ms' field.
#     Otherwise LS will be reported as None.

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Iterable

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError

# ------------------------------------------------------------
# CONFIG (aligns with llm_enrich_drugbank.py)
# ------------------------------------------------------------

S3_ENDPOINT = "http://localhost:9000"
BUCKET = "bronze"
GOLD_PREFIX = "gold/drugbank_llm/"

MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minioStrongPass123"

# Name of the Groq model; used for reporting only
CURRENT_MODEL_NAME = "LLaMA-3.1 (Groq)"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
# Pydantic schema for LLM outputs (same as llm_enrich_drugbank)
# ------------------------------------------------------------

class DrugEnrichment(BaseModel):
    summary: str = Field(..., description="2–3 sentence summary of the drug.")
    primary_indications: List[str] = Field(default_factory=list)
    mechanism_summary: str = Field(..., description="Mechanism description.")
    safety_notes: str = Field(..., description="Safety information.")
    tags: List[str] = Field(default_factory=list)

# ------------------------------------------------------------
# S3 helpers
# ------------------------------------------------------------

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

def list_llm_objects(s3):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=GOLD_PREFIX)
    return resp.get("Contents", [])

def get_latest_llm_key(s3) -> str:
    objects = list_llm_objects(s3)
    if not objects:
        raise RuntimeError(
            f"No LLM JSONL outputs found under s3://{BUCKET}/{GOLD_PREFIX}"
        )
    # choose the most recently modified
    latest = max(objects, key=lambda o: o["LastModified"])
    logging.info(f"Using latest LLM file: s3://{BUCKET}/{latest['Key']}")
    return latest["Key"]

def load_jsonl_from_s3(s3, key: str) -> List[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8")
    except ClientError as e:
        raise RuntimeError(f"Error reading s3://{BUCKET}/{key}: {e}")

    records: List[Dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping invalid JSONL line: {e}")
    logging.info(f"Loaded {len(records)} LLM-enriched records from JSONL.")
    return records

# ------------------------------------------------------------
# Text / token helpers
# ------------------------------------------------------------

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return TOKEN_RE.findall(text)

def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

# ------------------------------------------------------------
# Metrics dataclass
# ------------------------------------------------------------

@dataclass
class LLMRunMetrics:
    model_name: str
    num_records: int
    jsvs: float
    smr: float
    field_completeness_before: Dict[str, float]
    field_completeness_after: Dict[str, float]
    hdm: float
    hdm_threshold: float
    fce_mean_jaccard: float
    latency_ms_mean: float | None
    throughput_rec_per_min: float | None

# ------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------

def evaluate_llm_records(records: List[Dict[str, Any]]) -> LLMRunMetrics:
    if not records:
        raise ValueError("No records to evaluate.")

    total = len(records)

    # 4.4.1 JSON Structural Validity Score (JSVS)
    valid_json = 0

    # 4.4.2 Schema Match Rate (Pydantic validation)
    schema_ok = 0

    # field completeness (before = raw, after = llm_enriched)
    # We map internal LLM fields to raw pre-enrichment fields.
    FIELD_MAP = {
        "mechanism_summary": "mechanism",   # "Mechanism"
        "safety_notes": "toxicity",         # "Side effects / safety"
        "primary_indications": "indication"
    }
    before_counts = {k: 0 for k in FIELD_MAP.keys()}
    after_counts = {k: 0 for k in FIELD_MAP.keys()}

    # 4.4.3 / 4.4.6 Hallucination (HDM) + FCE (keyword Jaccard)
    jaccards: List[float] = []
    HDM_THRESHOLD = 0.10  # below this Jaccard we treat as potential hallucination
    low_sim_count = 0

    # 4.4.4 Latency Score, 4.4.5 Throughput
    latencies: List[float] = []
    timestamps: List[datetime] = []

    for rec in records:
        raw = rec.get("raw", {})
        llm = rec.get("llm_enriched", {})

        # Structural + schema validity (re-validate via Pydantic)
        try:
            _ = DrugEnrichment(**llm)
            valid_json += 1
            schema_ok += 1
        except ValidationError as e:
            logging.warning(f"Schema/JSON validation failed for a record: {e}")

        # field completeness before/after
        for llm_field, raw_field in FIELD_MAP.items():
            raw_val = (raw or {}).get(raw_field)
            llm_val = (llm or {}).get(llm_field)

            if raw_val not in (None, "", [], {}):
                before_counts[llm_field] += 1
            if llm_val not in (None, "", [], {}):
                after_counts[llm_field] += 1

        # FCE & HDM via Jaccard similarity
                # Safely collect raw text parts (coerce None -> "")
        raw_text_parts = [
            str((raw or {}).get("description") or ""),
            str((raw or {}).get("indication") or ""),
            str((raw or {}).get("mechanism") or ""),
            str((raw or {}).get("toxicity") or ""),
        ]
        # Safely collect LLM text parts
        llm_text_parts = [
            str((llm or {}).get("summary") or ""),
            str((llm or {}).get("mechanism_summary") or ""),
            str((llm or {}).get("safety_notes") or ""),
        ]

        raw_tokens = tokenize(" ".join(raw_text_parts))
        llm_tokens = tokenize(" ".join(llm_text_parts))

        sim = jaccard_similarity(raw_tokens, llm_tokens)
        jaccards.append(sim)
        if sim < HDM_THRESHOLD:
            low_sim_count += 1

        # latency + timestamps
        metrics = rec.get("metrics") or {}
        lat_ms = metrics.get("latency_ms")
        if isinstance(lat_ms, (int, float)) and lat_ms >= 0:
            latencies.append(float(lat_ms))

        ts_str = rec.get("processed_at")
        if ts_str:
            try:
                timestamps.append(datetime.fromisoformat(ts_str))
            except Exception:
                pass

    jsvs = valid_json / total
    smr = schema_ok / total

    field_completeness_before = {
        f: before_counts[f] / total for f in FIELD_MAP.keys()
    }
    field_completeness_after = {
        f: after_counts[f] / total for f in FIELD_MAP.keys()
    }

    # HDM: percentage of records with low similarity
    hdm = low_sim_count / total
    fce_mean = sum(jaccards) / len(jaccards) if jaccards else 0.0

    latency_mean = (
        sum(latencies) / len(latencies)
        if latencies else None
    )

    throughput = None
    if timestamps:
        t_min = min(timestamps)
        t_max = max(timestamps)
        dur_sec = max((t_max - t_min).total_seconds(), 1e-9)
        throughput = total / (dur_sec / 60.0)

    return LLMRunMetrics(
        model_name=CURRENT_MODEL_NAME,
        num_records=total,
        jsvs=jsvs,
        smr=smr,
        field_completeness_before=field_completeness_before,
        field_completeness_after=field_completeness_after,
        hdm=hdm,
        hdm_threshold=HDM_THRESHOLD,
        fce_mean_jaccard=fce_mean,
        latency_ms_mean=latency_mean,
        throughput_rec_per_min=throughput,
    )

# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def main():
    from pathlib import Path

    logging.info("Connecting to MinIO / S3...")
    s3 = get_s3_client()
    key = get_latest_llm_key(s3)
    records = load_jsonl_from_s3(s3, key)

    metrics = evaluate_llm_records(records)

    # Pretty print to console
    print("\n=== LLM Evaluation Metrics ===")
    print(f"Model:                 {metrics.model_name}")
    print(f"Records evaluated:     {metrics.num_records}")
    print(f"JSVS (JSON valid):     {metrics.jsvs:.3f}")
    print(f"SMR (schema match):    {metrics.smr:.3f}")
    print("Field completeness BEFORE enrichment:")
    for f, v in metrics.field_completeness_before.items():
        print(f"  {f}: {v:.2%}")
    print("Field completeness AFTER enrichment:")
    for f, v in metrics.field_completeness_after.items():
        print(f"  {f}: {v:.2%}")
    print(f"HDM (low-overlap rate): {metrics.hdm:.3%} "
          f"(threshold={metrics.hdm_threshold})")
    print(f"FCE (mean Jaccard):    {metrics.fce_mean_jaccard:.3f}")

    if metrics.latency_ms_mean is not None:
        print(f"Latency (mean, ms):    {metrics.latency_ms_mean:.1f}")
    else:
        print("Latency (mean, ms):    N/A (no latency_ms in JSONL; "
              "instrument llm_enrich_drugbank.py to record it)")

    if metrics.throughput_rec_per_min is not None:
        print(f"Throughput (rec/min):  {metrics.throughput_rec_per_min:.1f}")
    else:
        print("Throughput (rec/min):  N/A (no valid timestamps)")

    # Save to dq/out/gold/llm_eval_metrics.json so it fits your DQ report flow
    out_dir = (Path.cwd() / "dq" / "out" / "gold")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_eval_metrics.json"

    payload = {
        "model_name": metrics.model_name,
        "num_records": metrics.num_records,
        "jsvs": metrics.jsvs,
        "smr": metrics.smr,
        "field_completeness_before": metrics.field_completeness_before,
        "field_completeness_after": metrics.field_completeness_after,
        "hdm": metrics.hdm,
        "hdm_threshold": metrics.hdm_threshold,
        "fce_mean_jaccard": metrics.fce_mean_jaccard,
        "latency_ms_mean": metrics.latency_ms_mean,
        "throughput_rec_per_min": metrics.throughput_rec_per_min,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics JSON to: {out_path}")

if __name__ == "__main__":
    main()
