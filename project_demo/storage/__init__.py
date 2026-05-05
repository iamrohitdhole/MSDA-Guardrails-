"""
Storage layer for the MSDA Guardrails project.

Single source of truth for:
  - MinIO/S3 endpoint, credentials, bucket, and object keys
  - boto3 client construction
  - high-level read/write helpers that try MinIO first and fall back to local FS

Modules:
  config           — env-driven config; one place to override endpoint/bucket/keys
  minio_client     — thin boto3 wrapper; reachability check + bucket helpers
  artifact_store   — high-level helpers: read/write evidence parquet, evidence index
"""
