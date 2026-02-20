# project_demo/scripts/guardrails_validation.py

from typing import Dict, Tuple, List

REQUIRED_FIELDS = [
    "drug_name",
    "mechanism_of_action",
    "side_effects",
    "contraindications",
    "warnings",
    "safe_usage",
    "summary",
]

def validate_schema(record: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that the LLM output follows the expected schema.
    Returns (is_valid, errors).
    """
    errors = []

    # 1. Check required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing field: {field}")
        else:
            if record[field] is None:
                errors.append(f"Null value for field: {field}")
            elif isinstance(record[field], str) and not record[field].strip():
                errors.append(f"Empty string for field: {field}")

    # 2. Type checks for specific fields
    if "side_effects" in record and not isinstance(record["side_effects"], list):
        errors.append("side_effects must be a list of strings")

    if "contraindications" in record and not isinstance(record["contraindications"], list):
        errors.append("contraindications must be a list of strings")

    # 3. Optional: length checks
    if "summary" in record and isinstance(record["summary"], str):
        if len(record["summary"]) < 20:
            errors.append("summary is suspiciously short (<20 chars)")

    return (len(errors) == 0, errors)


def detect_hallucinations(record: Dict, source: Dict) -> Tuple[bool, List[str]]:
    """
    Very simple heuristic hallucination detector.
    In a real system, you might call external knowledge or more complex rules.
    Here we just flag obviously unsafe patterns.
    """
    warnings = []

    text_fields = []
    for key in ["mechanism_of_action", "warnings", "safe_usage", "summary"]:
        if key in record and isinstance(record[key], str):
            text_fields.append(record[key].lower())

    merged_text = " ".join(text_fields)

    # 1. Overconfident cures
    red_flag_phrases = [
        "cures all cancers",
        "has no side effects",
        "100% safe in all patients",
        "no risk",
    ]

    for phrase in red_flag_phrases:
        if phrase in merged_text:
            warnings.append(f"Potential hallucination: contains phrase '{phrase}'")

    # 2. Check for blatant contradictions with known groups
    # Example: if drug is "withdrawn" but summary says "widely used as first-line"
    if "groups" in source and isinstance(source["groups"], list):
        if "withdrawn" in [g.lower() for g in source["groups"]]:
            if "widely used" in merged_text or "first-line" in merged_text:
                warnings.append("Drug is withdrawn but LLM claims it's widely used.")

    return (len(warnings) == 0, warnings)


def validate_llm_output(record: Dict, source: Dict) -> Tuple[bool, List[str]]:
    """
    Combined validation: schema + simple hallucination check.
    Returns (is_valid, issues).
    """
    schema_ok, schema_errors = validate_schema(record)
    halluc_ok, halluc_warnings = detect_hallucinations(record, source)

    issues: List[str] = []
    if not schema_ok:
        issues.extend(schema_errors)
    if not halluc_ok:
        issues.extend(halluc_warnings)

    return (len(issues) == 0, issues)
