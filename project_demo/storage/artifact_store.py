"""
High-level artifact store: MinIO-first reads with local-FS fallback.

Resolution order for evidence parquet and evidence index:
  1. MinIO at s3://<bucket>/<key>     (primary)
  2. Local mirror under MSDA_LOCAL_FALLBACK_DIR/<key>
  3. Legacy local path (silver/silver_drugs_from_delta.parquet,
                        artifacts/demo2/evidence_index.pkl)

The runtime never raises if MinIO is unreachable; it returns the best
available local source and surfaces a `SourceInfo` so the UI can show
where the data came from.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from project_demo.storage.config import StorageConfig, load_config
from project_demo.storage.minio_client import (
    ensure_bucket,
    get_client,
    object_exists,
    list_prefix,
    check_reachable,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceInfo:
    """Tracks where a resolved artifact came from."""

    path: Path
    origin: str  # "minio", "local_cache", "legacy_local", "missing"
    detail: str  # human-readable detail for UI


# ── Internal helpers ──────────────────────────────────────────────────────────


def _download_object(cfg: StorageConfig, key: str, dest: Path) -> bool:
    """Download a single S3 object to dest. Returns True on success, False otherwise."""
    try:
        client = get_client(cfg)
        if not object_exists(client, cfg.s3_bucket, key):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(cfg.s3_bucket, key, str(dest))
        return True
    except Exception as e:
        logger.warning("Failed to download s3://%s/%s -> %s: %s", cfg.s3_bucket, key, dest, e)
        return False


def _download_prefix(cfg: StorageConfig, prefix: str, dest_dir: Path) -> int:
    """Download every object under prefix into dest_dir. Returns count downloaded."""
    try:
        client = get_client(cfg)
    except Exception as e:
        logger.warning("Cannot get S3 client: %s", e)
        return 0

    objs = list_prefix(client, cfg.s3_bucket, prefix)
    if not objs:
        return 0

    count = 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    for o in objs:
        key = o["Key"]
        rel = key[len(prefix):].lstrip("/")
        if not rel:
            continue
        out_path = dest_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(cfg.s3_bucket, key, str(out_path))
            count += 1
        except Exception as e:
            logger.warning("Skipping %s: %s", key, e)
    return count


def _is_parquet_dir_with_data(p: Path) -> bool:
    if not p.is_dir():
        return False
    return any(p.glob("*.parquet")) or any(p.glob("part-*"))


def _looks_like_parquet_file(p: Path) -> bool:
    return p.is_file() and p.suffix == ".parquet"


# ── Public API ────────────────────────────────────────────────────────────────


def get_evidence_parquet(cfg: Optional[StorageConfig] = None) -> SourceInfo:
    """
    Return a local Path that the chatbot can read with pandas.read_parquet.

    Tries MinIO → local cache → legacy local. Never raises.
    """
    cfg = cfg or load_config()
    key = cfg.evidence_parquet_key
    local_cache = cfg.local_evidence_parquet

    # 1) MinIO. Evidence parquet may be a single file OR a parquet directory
    #    (Spark output). Probe both shapes.
    reachable, _ = check_reachable(cfg, timeout_s=2.0)
    if reachable:
        try:
            client = get_client(cfg)
            # Single-file shape
            if object_exists(client, cfg.s3_bucket, key):
                local_cache.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(cfg.s3_bucket, key, str(local_cache))
                return SourceInfo(
                    path=local_cache,
                    origin="minio",
                    detail=f"s3://{cfg.s3_bucket}/{key}",
                )
            # Directory shape (e.g. evidence/drugs_evidence.parquet/part-*.parquet)
            dir_prefix = key.rstrip("/") + "/"
            objs = list_prefix(client, cfg.s3_bucket, dir_prefix)
            if objs:
                # Cache as a directory; remove any stale single-file with same name
                if local_cache.exists() and not local_cache.is_dir():
                    local_cache.unlink()
                n = _download_prefix(cfg, dir_prefix, local_cache)
                if n > 0:
                    return SourceInfo(
                        path=local_cache,
                        origin="minio",
                        detail=f"s3://{cfg.s3_bucket}/{dir_prefix} ({n} parts)",
                    )
        except Exception as e:
            logger.warning("MinIO evidence-parquet fetch failed; falling back: %s", e)

    # 2) Local cache from a previous successful fetch
    if _looks_like_parquet_file(local_cache) or _is_parquet_dir_with_data(local_cache):
        return SourceInfo(
            path=local_cache, origin="local_cache", detail=str(local_cache)
        )

    # 3) Legacy local path
    legacy = cfg.local_legacy_parquet
    if _looks_like_parquet_file(legacy) or _is_parquet_dir_with_data(legacy):
        return SourceInfo(path=legacy, origin="legacy_local", detail=str(legacy))

    return SourceInfo(path=local_cache, origin="missing", detail="no evidence parquet found")


def get_evidence_index(cfg: Optional[StorageConfig] = None) -> SourceInfo:
    """
    Return a local Path to the BM25 index pickle.

    Tries MinIO → local cache → legacy local. Never raises.
    """
    cfg = cfg or load_config()
    key = cfg.evidence_index_key
    local_cache = cfg.local_evidence_index

    reachable, _ = check_reachable(cfg, timeout_s=2.0)
    if reachable and _download_object(cfg, key, local_cache):
        return SourceInfo(
            path=local_cache, origin="minio", detail=f"s3://{cfg.s3_bucket}/{key}"
        )

    if local_cache.is_file():
        return SourceInfo(
            path=local_cache, origin="local_cache", detail=str(local_cache)
        )

    legacy = cfg.local_legacy_index
    if legacy.is_file():
        return SourceInfo(path=legacy, origin="legacy_local", detail=str(legacy))

    return SourceInfo(path=local_cache, origin="missing", detail="no evidence index found")


def upload_evidence_index(local_path: Path, cfg: Optional[StorageConfig] = None) -> str:
    """
    Upload a built BM25 index pickle to MinIO. Returns the s3 URI.
    Caller is expected to have built the index already.
    """
    cfg = cfg or load_config()
    if not local_path.is_file():
        raise FileNotFoundError(f"Index not found at {local_path}")
    client = get_client(cfg)
    ensure_bucket(client, cfg.s3_bucket)
    client.upload_file(str(local_path), cfg.s3_bucket, cfg.evidence_index_key)
    return f"s3://{cfg.s3_bucket}/{cfg.evidence_index_key}"


def upload_evidence_parquet(local_path: Path, cfg: Optional[StorageConfig] = None) -> str:
    """
    Upload an evidence parquet (single file or directory of part-*.parquet) to MinIO.
    Returns the s3 URI prefix that was written.
    """
    cfg = cfg or load_config()
    if not local_path.exists():
        raise FileNotFoundError(f"Parquet path not found at {local_path}")
    client = get_client(cfg)
    ensure_bucket(client, cfg.s3_bucket)
    key = cfg.evidence_parquet_key

    if local_path.is_file():
        client.upload_file(str(local_path), cfg.s3_bucket, key)
        return f"s3://{cfg.s3_bucket}/{key}"

    # Directory: upload each parquet part under the key as a prefix
    prefix = key.rstrip("/") + "/"
    n = 0
    for f in sorted(local_path.iterdir()):
        if not f.is_file():
            continue
        # Skip Spark/_SUCCESS, .crc, .DS_Store, etc.
        if f.name.startswith(".") or f.name.startswith("_"):
            continue
        client.upload_file(str(f), cfg.s3_bucket, prefix + f.name)
        n += 1
    if n == 0:
        raise RuntimeError(f"No parquet files found under {local_path}")
    return f"s3://{cfg.s3_bucket}/{prefix} ({n} parts)"


def status_summary(cfg: Optional[StorageConfig] = None) -> dict:
    """Lightweight status struct for the Streamlit sidebar."""
    cfg = cfg or load_config()
    reachable, msg = check_reachable(cfg, timeout_s=2.0)
    return {
        "endpoint": cfg.s3_endpoint,
        "bucket": cfg.s3_bucket,
        "reachable": reachable,
        "message": msg,
    }
