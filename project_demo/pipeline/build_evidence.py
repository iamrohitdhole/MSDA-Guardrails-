#!/usr/bin/env python3
"""
Build the canonical chatbot evidence dataset from the raw DrugBank XML.

Streams the 1.6 GB DrugBank XML, extracts ~17 structured/raw fields per drug,
filters records that have no clinical text at all, and writes a single-file
parquet plus a sibling _MANIFEST.json.

Output goes both to local disk (always) and to MinIO (if --upload-to-minio).

The evidence dataset is intentionally derived ONLY from raw DrugBank fields.
No LLM-generated summary, safety_notes, or other synthetic text is included
in the output — this is the chatbot's grounding source and must remain
faithful to the source corpus.

Usage:
    # Build from default XML location, write to local cache:
    python -m project_demo.pipeline.build_evidence

    # Limit to first N drugs for a smoke test:
    python -m project_demo.pipeline.build_evidence --limit 200

    # Upload result to MinIO when done:
    python -m project_demo.pipeline.build_evidence --upload-to-minio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from project_demo.storage.artifact_store import upload_evidence_parquet
from project_demo.storage.config import load_config
from project_demo.storage.minio_client import (
    check_reachable,
    ensure_bucket,
    get_client,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


NS = "{http://www.drugbank.ca}"


# Direct text-bearing children we extract verbatim from each <drug>.
# These are RAW DrugBank fields — no LLM-generated text is touched anywhere
# in this script. The CHUNK_FIELDS list in project_demo.rag.__init__ should
# stay in sync with this list (minus the structural fields).
RAW_TEXT_FIELDS = [
    "description",
    "indication",
    "mechanism-of-action",
    "pharmacodynamics",
    "toxicity",
    "absorption",
    "half-life",
    "metabolism",
    "protein-binding",
    "route-of-elimination",
]

# Container fields we flatten into a single concatenated text per drug.
# - drug-interactions: each child has <name>, <description>; we join the descriptions
# - food-interactions: each child has plain text; we join with newlines
CONTAINER_TEXT_FIELDS = [
    "drug-interactions",
    "food-interactions",
]


def _clean(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.replace("\r", "").strip()


def _list_texts(parent: Optional[ET.Element], child_tag: str) -> List[str]:
    if parent is None:
        return []
    out: List[str] = []
    for c in parent.findall(NS + child_tag):
        t = _clean(c.text)
        if t:
            out.append(t)
    return out


def _drug_interactions_text(elem: ET.Element) -> str:
    """
    Join every <drug-interactions><drug-interaction><description>...</description>
    into one paragraph. Each interaction description already mentions both drugs
    by name (e.g. "Warfarin may increase the anticoagulant activities of Lepirudin"),
    so the concatenated text is BM25-friendly.
    """
    container = elem.find(NS + "drug-interactions")
    if container is None:
        return ""
    pieces: List[str] = []
    for inter in container.findall(NS + "drug-interaction"):
        desc_el = inter.find(NS + "description")
        d = _clean(desc_el.text) if desc_el is not None else ""
        if d:
            pieces.append(d)
    return "\n".join(pieces)


def _food_interactions_text(elem: ET.Element) -> str:
    container = elem.find(NS + "food-interactions")
    if container is None:
        return ""
    return "\n".join(_list_texts(container, "food-interaction"))


def _categories(elem: ET.Element) -> List[str]:
    container = elem.find(NS + "categories")
    if container is None:
        return []
    out: List[str] = []
    for cat in container.findall(NS + "category"):
        name_el = cat.find(NS + "category")  # nested name node
        # Some DrugBank versions store category name directly under <category><category>name</category></category>
        if name_el is not None:
            t = _clean(name_el.text)
            if t:
                out.append(t)
        else:
            t = _clean(cat.text)
            if t:
                out.append(t)
    return out


def _primary_drug_id(elem: ET.Element) -> Optional[str]:
    primary = elem.find(f"{NS}drugbank-id[@primary='true']")
    if primary is not None and primary.text:
        return primary.text.strip()
    # Fallback: first drugbank-id
    first = elem.find(NS + "drugbank-id")
    if first is not None and first.text:
        return first.text.strip()
    return None


def extract_drug(elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Extract one canonical record from a <drug> element. Returns None if too sparse."""
    drug_id = _primary_drug_id(elem)
    name_el = elem.find(NS + "name")
    drug_name = _clean(name_el.text) if name_el is not None else ""
    if not drug_id or not drug_name:
        return None

    record: Dict[str, Any] = {"drug_id": drug_id, "drug_name": drug_name}

    # Raw text fields → snake_case
    for tag in RAW_TEXT_FIELDS:
        col = tag.replace("-", "_")
        sub = elem.find(NS + tag)
        record[col] = _clean(sub.text) if sub is not None else ""

    # Container text fields
    record["drug_interactions"] = _drug_interactions_text(elem)
    record["food_interactions"] = _food_interactions_text(elem)

    # Lists
    record["synonyms"] = _list_texts(elem.find(NS + "synonyms"), "synonym")
    record["groups"] = _list_texts(elem.find(NS + "groups"), "group")
    record["categories"] = _categories(elem)

    # Drop records with no clinical text whatsoever (would contribute zero chunks)
    text_blob_len = sum(
        len(record[c]) for c in (
            "description", "indication", "mechanism_of_action", "pharmacodynamics",
            "toxicity", "absorption", "half_life", "metabolism", "protein_binding",
            "route_of_elimination", "drug_interactions", "food_interactions",
        )
    )
    if text_blob_len == 0:
        return None

    return record


def iter_drugs(xml_path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    ctx = ET.iterparse(str(xml_path), events=("end",))
    yielded = 0
    for event, elem in ctx:
        if elem.tag != NS + "drug":
            continue
        rec = extract_drug(elem)
        # Free memory regardless
        elem.clear()
        if rec is None:
            continue
        yield rec
        yielded += 1
        if limit and yielded >= limit:
            return


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_parquet(records: List[Dict[str, Any]], dest: Path) -> None:
    import pandas as pd

    df = pd.DataFrame.from_records(records)
    # Stamp ingest_dt as UTC ISO
    df["ingest_dt"] = datetime.now(timezone.utc).isoformat()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False, compression="snappy")
    logger.info("Wrote %d rows to %s (%.1f MB)", len(df), dest, dest.stat().st_size / 1024 / 1024)


def write_manifest(
    dest: Path,
    *,
    build_id: str,
    source_xml_path: Path,
    source_xml_hash: str,
    row_count: int,
    s3_uri: Optional[str],
) -> None:
    manifest = {
        "build_id": build_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_xml_path": str(source_xml_path),
        "source_xml_hash": source_xml_hash,
        "row_count": row_count,
        "schema_version": "v2",
        "s3_uri": s3_uri,
        "evidence_basis": "raw_drugbank_only",  # explicit: no LLM-generated text
        "fields_in_chunk_basis": [
            "description", "indication", "mechanism_of_action", "pharmacodynamics",
            "toxicity", "absorption", "half_life", "metabolism", "protein_binding",
            "route_of_elimination", "drug_interactions", "food_interactions",
        ],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest -> %s", dest)


def _upload_manifest(local_path: Path) -> str:
    cfg = load_config()
    client = get_client(cfg)
    ensure_bucket(client, cfg.s3_bucket)
    # Place manifest next to the evidence parquet key
    parquet_dir = cfg.evidence_parquet_key.rsplit("/", 1)[0]
    manifest_key = f"{parquet_dir}/_MANIFEST.json"
    client.upload_file(str(local_path), cfg.s3_bucket, manifest_key)
    return f"s3://{cfg.s3_bucket}/{manifest_key}"


def main() -> int:
    cfg = load_config()
    default_xml = Path(
        "/Users/artemis/Downloads/MSDA-Guardrails-Project-main/project_demo/raw/full_database.xml"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=default_xml,
        help=f"Raw DrugBank XML path (default: {default_xml})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.local_evidence_parquet,
        help=f"Local output parquet (default: {cfg.local_evidence_parquet})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Local manifest path (default: alongside --out)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N drugs (smoke test)",
    )
    parser.add_argument(
        "--upload-to-minio",
        action="store_true",
        help="Upload the parquet + _MANIFEST.json to MinIO when done.",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip the SHA-256 of the source XML (faster but loses lineage proof).",
    )
    args = parser.parse_args()

    if not args.xml.is_file():
        print(f"[ERROR] XML not found at {args.xml}", file=sys.stderr)
        return 2

    manifest_path = args.manifest or args.out.with_name("_MANIFEST.json")

    logger.info("Reading XML from %s (%.1f MB)", args.xml, args.xml.stat().st_size / 1024 / 1024)

    if args.skip_hash:
        source_hash = "skipped"
    else:
        logger.info("Computing source XML SHA-256 (~30s for 1.6 GB)...")
        source_hash = sha256_file(args.xml)
        logger.info("source_xml_hash=%s", source_hash[:16] + "...")

    logger.info("Streaming-parsing DrugBank XML...")
    records: List[Dict[str, Any]] = []
    for i, rec in enumerate(iter_drugs(args.xml, limit=args.limit), start=1):
        records.append(rec)
        if i % 1000 == 0:
            logger.info("  parsed %d drug records", i)

    logger.info("Extracted %d drug records (filtered out drugs with no clinical text)", len(records))
    if not records:
        logger.error("No usable records extracted. Check the XML schema/namespace.")
        return 1

    write_parquet(records, args.out)

    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    s3_uri = None
    if args.upload_to_minio:
        ok, msg = check_reachable(cfg)
        if not ok:
            logger.error("Cannot upload — MinIO not reachable: %s", msg)
            return 1
        s3_uri = upload_evidence_parquet(args.out, cfg)
        logger.info("Uploaded evidence parquet -> %s", s3_uri)

    write_manifest(
        manifest_path,
        build_id=build_id,
        source_xml_path=args.xml,
        source_xml_hash=source_hash,
        row_count=len(records),
        s3_uri=s3_uri,
    )

    if args.upload_to_minio:
        manifest_uri = _upload_manifest(manifest_path)
        logger.info("Uploaded manifest -> %s", manifest_uri)

    print()
    print("Evidence dataset built.")
    print(f"  rows           : {len(records):,}")
    print(f"  local parquet  : {args.out}")
    print(f"  local manifest : {manifest_path}")
    if s3_uri:
        print(f"  MinIO parquet  : {s3_uri}")
    print()
    print("Next: rebuild the BM25 index from this dataset:")
    print(f"  python -m project_demo.rag.build_index --from-parquet {args.out}")
    print("  (omit --from-parquet to use MinIO-resolved source)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
