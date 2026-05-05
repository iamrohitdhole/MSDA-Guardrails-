#!/usr/bin/env python3
"""
Seed MinIO with the current local evidence artifacts so the chatbot can
read them as the primary source.

Usage:
    python -m project_demo.storage.seed                   # uses defaults
    python -m project_demo.storage.seed --parquet PATH    # override parquet source
    python -m project_demo.storage.seed --index PATH      # override index source
    python -m project_demo.storage.seed --skip-index      # parquet only
    python -m project_demo.storage.seed --skip-parquet    # index only

Reads MSDA_S3_* env vars for endpoint/bucket/credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from project_demo.storage.artifact_store import (
    upload_evidence_index,
    upload_evidence_parquet,
)
from project_demo.storage.config import load_config
from project_demo.storage.minio_client import check_reachable, ensure_bucket, get_client


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=cfg.local_legacy_parquet,
        help=f"Local parquet to upload as evidence (default: {cfg.local_legacy_parquet})",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=cfg.local_legacy_index,
        help=f"Local BM25 index to upload (default: {cfg.local_legacy_index})",
    )
    parser.add_argument("--skip-index", action="store_true", help="Skip uploading the index pickle")
    parser.add_argument("--skip-parquet", action="store_true", help="Skip uploading the parquet")
    args = parser.parse_args()

    ok, msg = check_reachable(cfg)
    if not ok:
        print(f"[ERROR] {msg}", file=sys.stderr)
        print(
            "Set MSDA_S3_ENDPOINT (default http://localhost:9000) and ensure MinIO is running.",
            file=sys.stderr,
        )
        return 2

    client = get_client(cfg)
    ensure_bucket(client, cfg.s3_bucket)
    print(f"[ok] bucket ready: {cfg.s3_bucket} @ {cfg.s3_endpoint}")

    if not args.skip_parquet:
        if not args.parquet.exists():
            print(f"[skip] parquet not found at {args.parquet}", file=sys.stderr)
        else:
            uri = upload_evidence_parquet(args.parquet, cfg)
            print(f"[ok] uploaded evidence parquet: {uri}")

    if not args.skip_index:
        if not args.index.is_file():
            print(f"[skip] index not found at {args.index}", file=sys.stderr)
        else:
            uri = upload_evidence_index(args.index, cfg)
            print(f"[ok] uploaded evidence index : {uri}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
