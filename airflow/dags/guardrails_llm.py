# project_demo/scripts/guardrails_llm.py

from typing import Dict, List, Tuple
import re

# These match your DrugEnrichment Pydantic model
REQUIRED_FIELDS = [
    "summary",
    "primary_indications",
    "mechanism_summary",
    "safety_notes",
    "tags",
]

# Naive text patterns that we treat as "hallucination red flags"
HALLUCINATION_PATTERNS = [
    r"cures all cancers",
    r"cures cancer",
    r"no side effects",
    r"100% safe",
    r"completely safe for everyone",
    r"works for all diseases",
    r"guaranteed to work for everyone",
]


def validate_required_fields(rec: Dict) -> Tuple[bool, List[str]]:
    """
    Check that all REQUIRED_FIELDS exist and are non-empty.

    Returns:
        (ok: bool, issues: list[str])
    """
    issues: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in rec:
            issues.append(f"Missing field: {field}")
            continue

        val = rec[field]

        if val is None:
            issues.append(f"Field {field} is None")
        elif isinstance(val, str) and not val.strip():
            issues.append(f"Field {field} is empty string")
        elif isinstance(val, list) and len(val) == 0:
            issues.append(f"Field {field} is empty list")

    return (len(issues) == 0), issues


def detect_hallucinations(rec: Dict) -> Tuple[bool, List[str]]:
    """
    Very naive hallucination detector based on forbidden phrases/patterns.

    Returns:
        (ok: bool, issues: list[str])
    """
    issues: List[str] = []
    text_chunks: List[str] = []

    # Collect all textual content from the enriched record
    for field in REQUIRED_FIELDS:
        if field not in rec:
            continue

        v = rec[field]
        if isinstance(v, str):
            text_chunks.append(v)
        elif isinstance(v, list):
            text_chunks.extend([str(x) for x in v])

    combined = " ".join(text_chunks).lower()

    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, combined):
            issues.append(f"Hallucination pattern matched: '{pattern}'")

    return (len(issues) == 0), issues


def compute_dq_score(
    field_ok: bool,
    field_issues: List[str],
    halluc_ok: bool,
    halluc_issues: List[str],
) -> float:
    """
    Compute a simple DQ (data quality) score in [0.0, 1.0].

    Start from 1.0 and subtract 0.1 per issue, plus extra penalties
    if field_ok or halluc_ok are false.
    """
    score = 1.0
    penalty_per_issue = 0.1

    total_issues = len(field_issues) + len(halluc_issues)
    score -= penalty_per_issue * total_issues

    if not field_ok:
        score -= 0.1
    if not halluc_ok:
        score -= 0.1

    # Clamp to [0, 1]
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    return score


def validate_enriched_record(rec: Dict, dq_threshold: float = 0.7) -> Dict:
    """
    Main entry point used by llm_enrich_drugbank.py.

    Args:
        rec: dict produced by DrugEnrichment Pydantic model
        dq_threshold: threshold for "valid" vs "low-quality"

    Returns:
        {
          "valid": bool,
          "dq_score": float,
          "field_issues": list[str],
          "hallucination_issues": list[str]
        }
    """
    field_ok, field_issues = validate_required_fields(rec)
    halluc_ok, halluc_issues = detect_hallucinations(rec)
    dq_score = compute_dq_score(field_ok, field_issues, halluc_ok, halluc_issues)

    is_valid = dq_score >= dq_threshold and field_ok and halluc_ok

    return {
        "valid": is_valid,
        "dq_score": dq_score,
        "field_issues": field_issues,
        "hallucination_issues": halluc_issues,
    }
