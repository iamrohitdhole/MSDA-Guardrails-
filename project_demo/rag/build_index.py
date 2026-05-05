#!/usr/bin/env python3
"""
Build the BM25 evidence index for the drug-information chatbot.

Default source: MinIO at s3://<MSDA_S3_BUCKET>/<MSDA_EVIDENCE_PARQUET_KEY>.
Falls back to a local mirror under MSDA_LOCAL_FALLBACK_DIR, then to the legacy
local parquet at silver/silver_drugs_from_delta.parquet.

Usage:
    # Default (MinIO-first; auto-falls back to local):
    python -m project_demo.rag.build_index

    # Explicit local parquet (bypasses MinIO):
    python -m project_demo.rag.build_index --from-parquet silver/silver_drugs_from_delta.parquet

    # Sample size for quick smoke test:
    python -m project_demo.rag.build_index --sample-limit 500

    # Build then upload index back to MinIO:
    python -m project_demo.rag.build_index --upload-to-minio

    # Legacy Spark/Delta path:
    python -m project_demo.rag.build_index --delta-path s3a://bronze/silver/drugs_clean/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_demo.rag import EVIDENCE_INDEX_PATH
from project_demo.rag.retriever import EvidenceRetriever
from project_demo.storage.artifact_store import (
    SourceInfo,
    get_evidence_parquet,
    upload_evidence_index,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Local data loaders (no Spark) ─────────────────────────────────────────────

def _load_parquet_local(path: str, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load drug records from a local parquet file or directory of parquet files."""
    import pandas as pd
    import glob as _glob

    p = Path(path)
    if p.is_dir():
        parts = sorted(_glob.glob(str(p / "part-*.parquet")))
        if not parts:
            parts = sorted(_glob.glob(str(p / "*.parquet")))
        if not parts:
            raise FileNotFoundError(f"No parquet files found in directory: {p}")
        df = pd.concat([pd.read_parquet(pp) for pp in parts], ignore_index=True)
    else:
        df = pd.read_parquet(path)

    logger.info("Loaded %d rows from parquet: %s", len(df), path)
    logger.info("Columns: %s", df.columns.tolist())

    if sample_limit:
        df = df.head(sample_limit)

    return df.fillna("").to_dict("records")


def _load_jsonl_local(path: str, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load drug records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Handle nested formats: {"raw": {...}, "llm_enriched": {...}}
                if "raw" in obj:
                    rec = dict(obj["raw"])
                    if "llm_enriched" in obj:
                        rec.update(obj["llm_enriched"])
                    records.append(rec)
                else:
                    records.append(obj)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line")

    logger.info("Loaded %d records from JSONL: %s", len(records), path)
    if sample_limit:
        records = records[:sample_limit]
    return records


# ── Spark / Delta path (legacy) ───────────────────────────────────────────────

def _build_from_delta(
    delta_path: str,
    output_path: Path,
    sample_limit: Optional[int] = None,
) -> None:
    """Build index from Spark Delta table (requires Spark + MinIO)."""
    from project_demo.scripts.spark_builder import build_spark
    from project_demo.rag.retriever import build_index_from_delta

    spark = build_spark("rag_build_index")
    try:
        build_index_from_delta(spark, delta_path, output_path, sample_limit=sample_limit)
    finally:
        spark.stop()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BM25 evidence index for the drug-information guardrail chatbot."
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-parquet",
        metavar="PATH",
        help="Load from a local parquet file or directory. Bypasses MinIO.",
    )
    source.add_argument(
        "--from-jsonl",
        metavar="PATH",
        help="Load from JSONL file (supports {raw: {...}, llm_enriched: {...}} format).",
    )
    source.add_argument(
        "--delta-path",
        metavar="S3A_PATH",
        help="Spark Delta table path (requires Spark + MinIO). Legacy.",
    )

    parser.add_argument(
        "--output-path", "-o",
        type=Path,
        default=EVIDENCE_INDEX_PATH,
        help=f"Output path for the index pickle. Default: {EVIDENCE_INDEX_PATH}",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of records (useful for quick testing).",
    )
    parser.add_argument(
        "--upload-to-minio",
        action="store_true",
        help="After building, upload the index pickle to MinIO under the configured key.",
    )

    args = parser.parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load records ──────────────────────────────────────────────────────────
    if args.from_parquet:
        records = _load_parquet_local(args.from_parquet, sample_limit=args.sample_limit)

    elif args.from_jsonl:
        records = _load_jsonl_local(args.from_jsonl, sample_limit=args.sample_limit)

    elif args.delta_path:
        # Legacy Spark / Delta path — only when explicitly requested
        logger.info("Using Spark Delta path: %s", args.delta_path)
        try:
            _build_from_delta(args.delta_path, args.output_path, sample_limit=args.sample_limit)
            print(f"\nIndex saved to {args.output_path}")
            print(f"Size: {args.output_path.stat().st_size / 1024:.1f} KB")
            if args.upload_to_minio:
                uri = upload_evidence_index(args.output_path)
                print(f"Uploaded to MinIO: {uri}")
            return 0
        except ImportError as e:
            logger.error(
                "Spark not available (%s). Use --from-parquet or --from-jsonl instead.", e
            )
            return 1

    else:
        # Default: MinIO-first → local cache → legacy local
        info: SourceInfo = get_evidence_parquet()
        if info.origin == "missing":
            logger.error(
                "No evidence parquet found. Tried MinIO and local fallbacks.\n"
                "Either upload one with: python -m project_demo.storage.seed\n"
                "Or pass --from-parquet PATH explicitly."
            )
            return 1
        logger.info("Loading evidence parquet from %s (%s)", info.origin, info.detail)
        records = _load_parquet_local(str(info.path), sample_limit=args.sample_limit)

    if not records:
        logger.error("No records loaded. Check the input path and format.")
        return 1

    logger.info("Building BM25 index from %d drug records...", len(records))

    retriever = EvidenceRetriever()
    retriever.build_index(records)
    retriever.save_index(args.output_path)

    size_kb = args.output_path.stat().st_size / 1024
    print(f"\nEvidence index built successfully.")
    print(f"  Records: {len(records):,}")
    print(f"  Output:  {args.output_path}")
    print(f"  Size:    {size_kb:.1f} KB")

    if args.upload_to_minio:
        try:
            uri = upload_evidence_index(args.output_path)
            print(f"  MinIO:   {uri}")
        except Exception as e:
            logger.error("Upload to MinIO failed: %s", e)
            return 1

    print(f"\nNext: run inference")
    print(f"  python -m project_demo.rag.run_inference --mode self_correcting --model mistral")
    return 0


if __name__ == "__main__":
    sys.exit(main())
