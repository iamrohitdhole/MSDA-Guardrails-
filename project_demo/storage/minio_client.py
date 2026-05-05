"""
Thin boto3 wrapper for MinIO/S3.

All callers go through `get_client()` so endpoint/credentials live in one place.
boto3 import is lazy so the rest of the project still works if it's not installed
(local-only fallback path).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from project_demo.storage.config import StorageConfig, load_config

logger = logging.getLogger(__name__)


def _import_boto3():
    try:
        import boto3
        from botocore.client import Config as BotoConfig
        from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

        return boto3, BotoConfig, (BotoCoreError, ClientError, EndpointConnectionError)
    except ImportError as e:
        raise RuntimeError(
            "boto3 is required for MinIO access. Install with: pip install boto3"
        ) from e


def get_client(cfg: Optional[StorageConfig] = None) -> Any:
    """Return a boto3 S3 client configured for MinIO (path-style, sigv4)."""
    cfg = cfg or load_config()
    boto3, BotoConfig, _ = _import_boto3()
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3_endpoint,
        aws_access_key_id=cfg.s3_access_key,
        aws_secret_access_key=cfg.s3_secret_key,
        region_name=cfg.s3_region,
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def check_reachable(cfg: Optional[StorageConfig] = None, timeout_s: float = 2.0) -> Tuple[bool, str]:
    """
    Probe MinIO with a low-cost call. Returns (ok, message).
    Never raises — used by the Streamlit sidebar to render status.
    """
    cfg = cfg or load_config()
    try:
        boto3, BotoConfig, exc_types = _import_boto3()
    except RuntimeError as e:
        return False, str(e)

    try:
        client = boto3.client(
            "s3",
            endpoint_url=cfg.s3_endpoint,
            aws_access_key_id=cfg.s3_access_key,
            aws_secret_access_key=cfg.s3_secret_key,
            region_name=cfg.s3_region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=timeout_s,
                read_timeout=timeout_s,
                retries={"max_attempts": 1},
            ),
        )
        client.list_buckets()
        return True, f"MinIO reachable at {cfg.s3_endpoint}"
    except exc_types as e:
        return False, f"MinIO not reachable at {cfg.s3_endpoint}: {e}"
    except Exception as e:
        return False, f"MinIO probe failed: {e}"


def ensure_bucket(client: Any, bucket: str) -> None:
    """Create bucket if missing. No-op if it already exists."""
    _, _, exc_types = _import_boto3()
    try:
        client.head_bucket(Bucket=bucket)
        return
    except exc_types:
        pass
    try:
        client.create_bucket(Bucket=bucket)
        logger.info("Created bucket: %s", bucket)
    except exc_types as e:
        msg = str(e)
        if "BucketAlreadyOwnedByYou" in msg or "BucketAlreadyExists" in msg:
            return
        raise


def object_exists(client: Any, bucket: str, key: str) -> bool:
    _, _, exc_types = _import_boto3()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except exc_types:
        return False


def list_prefix(client: Any, bucket: str, prefix: str) -> list[dict]:
    _, _, exc_types = _import_boto3()
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return resp.get("Contents", []) or []
    except exc_types as e:
        logger.warning("list_objects_v2 failed for %s/%s: %s", bucket, prefix, e)
        return []
