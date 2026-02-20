import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Iterable, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

# Try to import Groq. If unavailable (e.g. inside some containers),
# we will fall back to a simple, non-LLM enrichment.
try:
    from groq import Groq  # type: ignore
    HAS_GROQ = True
except Exception:  # ImportError or any other startup issue
    Groq = None  # type: ignore
    HAS_GROQ = False
    logging.warning(
        "[llm_enrich_drugbank] groq package not available. "
        "Will use fallback enrichment (no external LLM calls)."
    )

# ---------------------------------------
# Pydantic validation model
# ---------------------------------------
from pydantic import BaseModel, Field, ValidationError


class DrugEnrichment(BaseModel):
    """
    Structured LLM enrichment for a single DrugBank record.
    All fields are required in the schema; the model will enforce
    data types and give us a safe, validated object.
    """

    summary: str = Field(..., description="2–3 sentence summary of the drug.")
    primary_indications: List[str] = Field(
        default_factory=list,
        description="List of 2–4 key conditions or situations where the drug is used.",
    )
    mechanism_summary: str = Field(
        ...,
        description="Short description of mechanism of action in plain language.",
    )
    safety_notes: str = Field(
        ...,
        description="Critical safety information, contraindications, or monitoring.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="3–6 short tags to help quickly categorize the drug.",
    )


# ============================================================
# CONFIG
# ============================================================

# IMPORTANT:
# Use an environment variable so the same script works from:
#   - your host (S3_ENDPOINT=http://localhost:9000)
#   - inside Airflow (S3_ENDPOINT=http://minio:9000)
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")

BUCKET = "bronze"

SILVER_PREFIX = "silver/drugs_clean/"
GOLD_PREFIX = "gold/drugbank_llm/"

# Credentials for MinIO (same as the rest of the project)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioStrongPass123")

# Groq
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Current recommended Groq model

# Batch / throttling
MAX_RECORDS = 25          # limit per run to control token cost
LOG_EVERY = 5
CALL_INTERVAL = 0.8       # seconds between calls
MAX_RETRIES = 5

# Data-quality thresholds
DQ_WARN_THRESHOLD = 0.85  # warn if overall DQ is below this

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


# ============================================================
# MINIO / S3 HELPERS
# ============================================================


def get_s3_client():
    logging.info(f"[llm_enrich_drugbank] Using MinIO/S3 endpoint: {S3_ENDPOINT}")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def list_sample_objects(s3):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=SILVER_PREFIX)
    return resp.get("Contents", [])


def get_latest_sample_key(s3):
    objects = list_sample_objects(s3)
    if not objects:
        raise RuntimeError("No silver DrugBank samples found.")

    latest = max(objects, key=lambda o: o["LastModified"])
    logging.info(f"Using latest DrugBank sample: s3://{BUCKET}/{latest['Key']}")
    return latest["Key"]


def load_records_from_s3(s3, key: str) -> Iterable[Dict[str, Any]]:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    for line in obj["Body"].read().decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def write_jsonl_to_s3(s3, key: str, records: List[Dict[str, Any]]):
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode("utf-8"))
    logging.info(f"Wrote {len(records)} enriched records to s3://{BUCKET}/{key}")


# ============================================================
# GROQ HELPERS
# ============================================================


def get_groq_client() -> Optional["Groq"]:
    """
    Return a Groq client if available and configured.
    If not, returns None so the caller can fall back gracefully.
    """
    if not HAS_GROQ:
        logging.warning(
            "[llm_enrich_drugbank] groq library not available. "
            "get_groq_client() will return None and use fallback enrichment."
        )
        return None

    api_key = os.getenv(GROQ_API_KEY_ENV)
    if not api_key:
        logging.warning(
            f"[llm_enrich_drugbank] {GROQ_API_KEY_ENV} not set – "
            "will use fallback (no external LLM calls)."
        )
        return None

    return Groq(api_key=api_key)


def build_prompt(record: Dict[str, Any]) -> str:
    """
    Build a strongly-constrained prompt to reduce empty fields and
    improve data quality. We explicitly tell the model:

      * Never leave fields empty.
      * If truly unknown, write 'Not specified'.
      * Provide 2–4 indications and 3–6 tags.
    """
    name = record.get("name", "Unknown drug")
    description = record.get("description", "N/A")
    indication = record.get("indication", "N/A")
    mechanism = record.get("mechanism_of_action", "N/A")
    pharmacodynamics = record.get("pharmacodynamics", "N/A")
    cas = record.get("cas_number", "N/A")
    groups = record.get("groups") or []
    synonyms = record.get("synonyms") or []

    if isinstance(groups, list):
        groups = ", ".join(groups)
    if isinstance(synonyms, list):
        synonyms = ", ".join(synonyms)

    drug_id = record.get("drugbank_id") or record.get("id") or "unknown_id"

    return f"""
You are helping to enrich DrugBank records for an enterprise guardrail system.

You MUST return ONLY a single valid JSON object that follows this exact schema
and field names (no markdown, no backticks, no explanations):

{{
  "summary": "string",
  "primary_indications": ["string", ...],
  "mechanism_summary": "string",
  "safety_notes": "string",
  "tags": ["string", ...]
}}

Hard requirements (very important):
- Every field is REQUIRED. Never return an empty string "" or an empty list [].
- If the source data does not mention something, write a short phrase like "Not specified".
- "primary_indications": include 2–4 specific conditions or use-cases, in plain language.
- "tags": include 3–6 short, lower-case tags like "antibiotic", "cardiology", "oral", etc.
- Keep all text factual and grounded in the provided drug data. If unsure, say "Not specified" rather than guessing.

Drug Data:
- Drug ID: {drug_id}
- Name: {name}
- CAS: {cas}
- Groups: {groups}
- Synonyms: {synonyms}

- Indication: {indication}
- Mechanism of Action: {mechanism}
- Pharmacodynamics: {pharmacodynamics}
- Description: {description}

Now output the JSON object ONLY, with no extra text.
""".strip()


def fallback_enrichment(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback enrichment used when Groq is not available.
    This does NOT call any external LLM and just builds a simple
    structured summary from existing DrugBank fields.
    """
    name = record.get("name", "Unknown drug")
    indication = record.get("indication") or record.get("description") or ""
    mechanism = record.get("mechanism_of_action") or record.get("pharmacodynamics") or ""
    groups = record.get("groups") or []
    if isinstance(groups, list):
        groups_str = ", ".join(groups)
    else:
        groups_str = str(groups)

    summary_bits = []
    if name:
        summary_bits.append(f"{name} is a drug in the DrugBank dataset.")
    if groups_str:
        summary_bits.append(f"It belongs to the following group(s): {groups_str}.")
    if indication:
        summary_bits.append(f"Typical indications include: {indication[:200]}")

    if not summary_bits:
        summary_bits.append("Information not specified in the source record.")

    return DrugEnrichment(
        summary=" ".join(summary_bits),
        primary_indications=[
            indication[:120] or "Not specified"
        ],
        mechanism_summary=mechanism[:240] or "Mechanism not specified.",
        safety_notes="Safety information not specified in the source record.",
        tags=["not-specified"],
    ).model_dump()


def call_groq(client: Optional["Groq"], prompt: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Groq with strict JSON output + Pydantic validation.
    If Groq is unavailable, uses the fallback enrichment instead.

    Returns a dict that conforms to the DrugEnrichment schema.
    """
    if client is None:
        # No Groq in this environment – deterministic fallback
        return fallback_enrichment(record)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(CALL_INTERVAL)

            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful assistant that outputs ONLY valid JSON "
                            "following the specified schema. "
                            "No markdown, no explanations, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.05,
            )

            content = resp.choices[0].message.content.strip()

            # Attempt to parse JSON
            raw_json = json.loads(content)

            # Validate with Pydantic (v2 style)
            validated = DrugEnrichment(**raw_json)
            return validated.model_dump()

        except Exception as e:
            msg = str(e)

            if attempt < MAX_RETRIES:
                wait = attempt * 2
                logging.warning(
                    f"Groq / validation error (attempt {attempt}/{MAX_RETRIES}): {msg}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                # Last resort: return fallback JSON
                logging.error(
                    "[llm_enrich_drugbank] FAILED after all retries. "
                    "Using fallback JSON enrichment."
                )
                return fallback_enrichment(record)


# ============================================================
# DATA-QUALITY SCORING
# ============================================================


def calculate_dq_score(
    enrichment: DrugEnrichment,
    source_record: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[str], List[str]]:
    """
    Compute a simple data-quality (DQ) score in [0, 1].

    We reward:
      - Non-empty fields.
      - Lists with at least a few useful items.

    We flag:
      - Empty strings / lists.
      - Very generic "Not specified" text across all fields.

    This is intentionally simple and explainable for your report.
    """
    field_issues: List[str] = []
    hallucination_issues: List[str] = []

    score = 1.0

    # --- Field completeness ---
    if not enrichment.summary.strip():
        score -= 0.25
        field_issues.append("Field summary is empty string")

    if not enrichment.primary_indications:
        score -= 0.15
        field_issues.append("Field primary_indications is empty list")
    elif len(enrichment.primary_indications) < 2:
        score -= 0.05
        field_issues.append("Field primary_indications has fewer than 2 items")

    if not enrichment.mechanism_summary.strip():
        score -= 0.20
        field_issues.append("Field mechanism_summary is empty string")

    if not enrichment.safety_notes.strip():
        score -= 0.20
        field_issues.append("Field safety_notes is empty string")

    if not enrichment.tags:
        score -= 0.15
        field_issues.append("Field tags is empty list")
    elif len(enrichment.tags) < 3:
        score -= 0.05
        field_issues.append("Field tags has fewer than 3 items")

    # --- Very generic "Not specified" across the board ---
    # (soft penalty; we prefer explicit coverage but avoid over-penalising.)
    generic_phrases = ["not specified", "information not available"]
    generic_hits = 0
    for text_field in [
        enrichment.summary,
        enrichment.mechanism_summary,
        enrichment.safety_notes,
    ]:
        lowered = text_field.lower()
        if any(p in lowered for p in generic_phrases):
            generic_hits += 1

    if generic_hits >= 2:
        score -= 0.1
        field_issues.append(
            "Multiple text fields rely on generic 'not specified' style wording"
        )

    # --- Extremely lightweight hallucination check (optional) ---
    if source_record is not None:
        name = (source_record.get("name") or "").strip()
        if name:
            combined_text = " ".join(
                [
                    enrichment.summary,
                    enrichment.mechanism_summary,
                    enrichment.safety_notes,
                ]
            ).lower()
            # If the drug name literally never appears anywhere, we add a small warning.
            if name.lower() not in combined_text:
                hallucination_issues.append(
                    "Drug name from source does not appear in enrichment text"
                )
                score -= 0.05

    # Clamp score to [0.0, 1.0]
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    return score, field_issues, hallucination_issues


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():
    s3 = get_s3_client()
    groq_client = get_groq_client()

    sample_key = get_latest_sample_key(s3)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_key = f"{GOLD_PREFIX}drugbank_llm_{ts}.jsonl"
    logging.info(f"Will write enriched output to s3://{BUCKET}/{out_key}")

    enriched_records: List[Dict[str, Any]] = []
    count = 0

    for record in load_records_from_s3(s3, sample_key):
        count += 1
        if MAX_RECORDS and count > MAX_RECORDS:
            break

        prompt = build_prompt(record)
        llm_json = call_groq(groq_client, prompt, record)

        # At this point llm_json should conform to DrugEnrichment schema.
        # We re-instantiate the model so we can compute DQ in a structured way.
        try:
            enrichment_obj = DrugEnrichment(**llm_json)
        except ValidationError as e:
            logging.error(
                f"ValidationError after call_groq for record #{count}: {e}. "
                "Falling back to minimal enrichment."
            )
            enrichment_obj = DrugEnrichment(
                summary=llm_json.get("summary") or "Information not available.",
                primary_indications=llm_json.get("primary_indications") or [],
                mechanism_summary=llm_json.get("mechanism_summary")
                or "Mechanism not available.",
                safety_notes=llm_json.get("safety_notes")
                or "Safety information not available.",
                tags=llm_json.get("tags") or [],
            )

        dq_score, field_issues, hallucination_issues = calculate_dq_score(
            enrichment_obj, record
        )

        if dq_score < DQ_WARN_THRESHOLD:
            logging.warning(
                f"Low DQ for drug record #{count}: "
                f"dq_score={dq_score:.2f}, "
                f"field_issues={field_issues}, "
                f"hallucination_issues={hallucination_issues}"
            )
        else:
            logging.info(
                f"DQ OK for drug record #{count}: dq_score={dq_score:.2f}"
            )

        enriched_records.append(
            {
                "raw": record,
                "llm_enriched": enrichment_obj.model_dump(),
                "dq_score": dq_score,
                "dq_field_issues": field_issues,
                "dq_hallucination_issues": hallucination_issues,
                "processed_at": datetime.utcnow().isoformat(),
            }
        )

        if count % LOG_EVERY == 0:
            logging.info(f"Processed {count} records...")

    if not enriched_records:
        logging.warning("No records were enriched. Nothing to write.")
        return

    write_jsonl_to_s3(s3, out_key, enriched_records)


if __name__ == "__main__":
    main()
