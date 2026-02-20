# config_io.py
from pathlib import Path
import os

# When reading inputs, prefer S3 if True; otherwise local (writes always go to BOTH)
PREFER_S3_READ = False

# MinIO connection (env vars can override)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioStrongPass123")

# Your MinIO bucket (you already have "bronze")
BUCKET  = os.getenv("MSDA_BUCKET", "bronze")
BASE_S3A = f"s3a://{BUCKET}"

# Local project root
ROOT = Path(__file__).resolve().parents[1]

# Canonical Delta locations
SILVER_DELTA_LOCAL = ROOT / "lake" / "silver"   / "silver_good_delta"
FEAT_DELTA_LOCAL   = ROOT / "lake" / "feature"  / "drugs_features_delta"

SILVER_DELTA_S3 = f"{BASE_S3A}/silver/silver_good_delta"
FEAT_DELTA_S3   = f"{BASE_S3A}/feature/drugs_features_delta"
