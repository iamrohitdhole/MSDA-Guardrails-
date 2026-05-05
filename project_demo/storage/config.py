"""
Storage configuration — env-driven, with safe local-dev defaults.

Environment variables (all optional):
  MSDA_S3_ENDPOINT           e.g. http://localhost:9000  or  http://minio:9000
  MSDA_S3_BUCKET             default: msda-drugbank
  MSDA_S3_ACCESS_KEY         default: minio
  MSDA_S3_SECRET_KEY         default: minioStrongPass123
  MSDA_S3_REGION             default: us-east-1
  MSDA_EVIDENCE_PARQUET_KEY  default: evidence/drugs_evidence.parquet
  MSDA_EVIDENCE_INDEX_KEY    default: evidence/drugs_evidence_index.pkl
  MSDA_LOCAL_FALLBACK_DIR    default: <repo-root>/data
  MSDA_LOCAL_LEGACY_PARQUET  default: <repo-root>/silver/silver_drugs_from_delta.parquet
                             (legacy local source used as last-resort fallback)
  MSDA_LOCAL_LEGACY_INDEX    default: <repo-root>/artifacts/demo2/evidence_index.pkl
                             (legacy local index used as last-resort fallback)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    val = os.getenv(name, "").strip()
    return val if val else default


@dataclass(frozen=True)
class StorageConfig:
    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str
    evidence_parquet_key: str
    evidence_index_key: str
    local_fallback_dir: Path
    local_legacy_parquet: Path
    local_legacy_index: Path

    @property
    def local_evidence_parquet(self) -> Path:
        """Mirror of <bucket>/<evidence_parquet_key> on local disk."""
        return self.local_fallback_dir / self.evidence_parquet_key

    @property
    def local_evidence_index(self) -> Path:
        """Mirror of <bucket>/<evidence_index_key> on local disk."""
        return self.local_fallback_dir / self.evidence_index_key


def load_config() -> StorageConfig:
    return StorageConfig(
        s3_endpoint=_env("MSDA_S3_ENDPOINT", "http://localhost:9000"),
        s3_bucket=_env("MSDA_S3_BUCKET", "msda-drugbank"),
        s3_access_key=_env("MSDA_S3_ACCESS_KEY", "minio"),
        s3_secret_key=_env("MSDA_S3_SECRET_KEY", "minioStrongPass123"),
        s3_region=_env("MSDA_S3_REGION", "us-east-1"),
        evidence_parquet_key=_env(
            "MSDA_EVIDENCE_PARQUET_KEY", "evidence/drugs_evidence.parquet"
        ),
        evidence_index_key=_env(
            "MSDA_EVIDENCE_INDEX_KEY", "evidence/drugs_evidence_index.pkl"
        ),
        local_fallback_dir=Path(_env("MSDA_LOCAL_FALLBACK_DIR", str(_REPO_ROOT / "data"))),
        local_legacy_parquet=Path(
            _env(
                "MSDA_LOCAL_LEGACY_PARQUET",
                str(_REPO_ROOT / "silver" / "silver_drugs_from_delta.parquet"),
            )
        ),
        local_legacy_index=Path(
            _env(
                "MSDA_LOCAL_LEGACY_INDEX",
                str(_REPO_ROOT / "artifacts" / "demo2" / "evidence_index.pkl"),
            )
        ),
    )
