"""
Guardrail Decision Logic — layered safety checks for the drug-information chatbot.

Layer 1 (shared, all modes): Input scope / safety policy checks.
Layer 2 (evidence-based):    JSON parse → schema → grounding → coverage.
Layer 3 (output):            Binary ALLOW / BLOCK with safe refusal message.

No WARN state. Final decision is strictly ALLOW or BLOCK.
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from project_demo.rag import GROUNDING_THRESHOLD_DEFAULT, MIN_COVERAGE_RATIO
from project_demo.rag.retriever import tokenize, retrieve_evidence


# ── DrugEnrichment schema ─────────────────────────────────────────────────────

class DrugEnrichment(BaseModel):
    # Optional focused factoid answer. Populated by the model when the user's
    # query is asking about a specific field (half-life, side effects, etc.)
    # and the retrieved evidence contains that field. When non-empty and
    # non-trivial, the renderer leads the response with this string.
    direct_answer: Optional[str] = Field(
        default=None,
        description="Optional 1–2 sentence direct factoid answer to the user's specific question.",
    )
    summary: str = Field(..., description="2–3 sentence summary of the drug.")
    primary_indications: List[str] = Field(
        default_factory=list,
        description="2–4 key conditions or situations where the drug is used.",
    )
    mechanism_summary: str = Field(..., description="Mechanism of action in plain language.")
    safety_notes: str = Field(..., description="Critical safety information or contraindications.")
    tags: List[str] = Field(default_factory=list, description="3–6 short lower-case tags.")


# ── Safe refusal messages ─────────────────────────────────────────────────────

SAFE_REFUSAL_TEMPLATE = (
    "I'm sorry, I cannot provide {reason}. "
    "For accurate medical information, please consult a qualified healthcare professional, "
    "pharmacist, or refer to authoritative drug information resources."
)

SAFE_REFUSAL_DEFAULT = (
    "I'm sorry, I cannot provide a verified answer to this question based on the available evidence. "
    "The information may be outside the supported drug-information scope, or the evidence is "
    "insufficient to provide a grounded response. Please consult a healthcare professional."
)

OUT_OF_SCOPE_QUERY_MESSAGE = (
    "I can only answer drug-related questions supported by the knowledge base. "
    "Please ask about a drug's uses, side effects, interactions, contraindications, or mechanism. "
    "For example: 'What is aspirin used for?' or 'What are the side effects of ibuprofen?'"
)


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — INPUT SCOPE / SAFETY POLICY CHECKS
# ════════════════════════════════════════════════════════════════════════════

class ScopeCheckResult(BaseModel):
    blocked: bool
    rule_tag: Optional[str] = None
    refusal_message: Optional[str] = None
    reason: Optional[str] = None


# ── Drug-relatedness check ───────────────────────────────────────────────────

# Keywords that indicate drug-related queries
_DRUG_KEYWORDS = re.compile(
    r"\b(drug|medication|medicine|pharmaceutical|pill|tablet|capsule|injection|"
    r"dose|dosage|side effects?|adverse effects?|contraindications?|indications?|"
    r"mechanism|mechanism of action|interactions?|efficacy|safety|toxicity|"
    r"aspirin|ibuprofen|warfarin|metformin|lisinopril|omeprazole|acetaminophen|"
    r"acetylsalicylic|paracetamol|ranitidine|sertraline|amoxicillin|"
    r"naproxen|cetirizine|loratadine|atorvastatin|simvastatin|"
    r"antacids?|antibiotics?|antidepressants?|antihistamines?|antihypertensives?|"
    r"anti-inflammatories?|analgesics?|anticoagulants?|vasodilators?|beta.?blockers?|"
    r"statins?|ace.?inhibitors?|ssris?|nsaids?|otc|over.the.counter|prescriptions?|"
    r"generic|brand names?|active ingredients?|formulations?|bioavailability|"
    r"pharmacokinetics|pharmacodynamics|metabolism|elimination|"
    r"half.?lives?|peak concentrations?|onset|duration|therapeutic ranges?)\b",
    re.IGNORECASE | re.VERBOSE
)


def is_drug_related_query(query: str) -> bool:
    """
    Check if the query is fundamentally drug-related based on keyword presence.
    Returns True if query contains drug-related terminology.
    """
    return bool(_DRUG_KEYWORDS.search(query))


# Ordered patterns: (compiled_regex, rule_tag, short_reason, safe_refusal)
_SCOPE_RULES: List[Tuple[re.Pattern, str, str, str]] = [
    # Diagnosis requests
    (
        re.compile(
            r"\b(diagnose|diagnosis|do i have|am i sick|am i infected|"
            r"what disease do i|test me for|check if i have|"
            r"is it cancer|could it be|what is wrong with me)\b",
            re.I,
        ),
        "OUT_OF_SCOPE_MEDICAL_ADVICE",
        "medical diagnosis",
        SAFE_REFUSAL_TEMPLATE.format(reason="medical diagnoses"),
    ),
    # Prescribing advice
    (
        re.compile(
            r"\b(prescribe me|write me a prescription|get me a prescription|"
            r"should i be prescribed|can you prescribe|"
            r"prescribing advice|prescribing recommendation)\b",
            re.I,
        ),
        "SAFETY_POLICY_BLOCK",
        "prescribing advice",
        SAFE_REFUSAL_TEMPLATE.format(reason="prescribing advice"),
    ),
    # Patient-specific dosing
    (
        re.compile(
            r"\b(how much.{0,30}should i take|my dosage|what dose\b.{0,40}should i|dosage for me|"
            r"my personal dose|dose for my weight|dose for my age|"
            r"how many mg should i|what is my dose|"
            r"dose adjustment for me|taper for me|titrate for me|"
            r"should i take\b.{0,40}\b(daily|weekly|per day)\b)\b",
            re.I,
        ),
        "SAFETY_POLICY_BLOCK",
        "patient-specific dosing",
        SAFE_REFUSAL_TEMPLATE.format(reason="patient-specific dosing advice"),
    ),
    # Treatment planning / personalized medical advice
    (
        re.compile(
            r"\b(treat my|cure my|treat me for|cure me of|"
            r"treatment plan for me|my treatment|should i stop taking|"
            r"switch my medication|change my medication|"
            r"is it safe for me to take|can i take .{0,30} with my)\b",
            re.I,
        ),
        "OUT_OF_SCOPE_MEDICAL_ADVICE",
        "personalized treatment planning",
        SAFE_REFUSAL_TEMPLATE.format(reason="personalized treatment plans"),
    ),
    # Overdose / self-harm
    (
        re.compile(
            r"\b(overdose|how to overdose|lethal dose|how much to kill|"
            r"suicide|self.harm|end my life)\b",
            re.I,
        ),
        "SAFETY_POLICY_BLOCK",
        "unsafe content",
        (
            "I'm sorry, I cannot help with that request. "
            "If you're in crisis, please call emergency services or a mental health helpline immediately."
        ),
    ),
]


def check_input_scope(query: str) -> ScopeCheckResult:
    """
    Check whether the query falls within the supported drug-information scope.

    Layer 1: Check if query is fundamentally drug-related.
    Layer 2: Check for unsafe/out-of-scope patterns (diagnosis, prescribing, patient-specific dosing, etc.)

    Blocks: non-drug queries, diagnosis, prescribing, patient-specific dosing, unsafe medical advice.
    Returns ScopeCheckResult with blocked=True and a safe refusal if out of scope.
    """
    # Layer 1: Reject queries that aren't drug-related at all
    if not is_drug_related_query(query):
        return ScopeCheckResult(
            blocked=True,
            rule_tag="OUT_OF_SCOPE_QUERY",
            refusal_message=OUT_OF_SCOPE_QUERY_MESSAGE,
            reason="non-drug-related query",
        )

    # Layer 2: Check for specific unsafe patterns
    for pattern, rule_tag, reason, refusal in _SCOPE_RULES:
        if pattern.search(query):
            return ScopeCheckResult(
                blocked=True,
                rule_tag=rule_tag,
                refusal_message=refusal,
                reason=reason,
            )
    return ScopeCheckResult(blocked=False)


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — EVIDENCE-BASED GUARDRAIL CHECKS
# ════════════════════════════════════════════════════════════════════════════

# ── CHECK 1: JSON Parse ───────────────────────────────────────────────────────

def check_json_parse(content: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Parse JSON from content string, stripping markdown fences if present."""
    try:
        content = content.strip()
        if content.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
        parsed = json.loads(content)
        return True, None, parsed
    except json.JSONDecodeError as e:
        return False, f"JSON_PARSE_ERROR: {e}", None


# ── CHECK 2: Schema Validation ────────────────────────────────────────────────

def check_schema(parsed: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate parsed dict against DrugEnrichment schema."""
    try:
        DrugEnrichment(**parsed)
        return True, []
    except ValidationError as e:
        issues = [
            f"SCHEMA_ERROR: {err.get('loc', ['unknown'])[0]} — {err.get('msg', 'unknown')}"
            for err in e.errors()
        ]
        return False, issues


# ── CHECK 3: Grounding Score (TF-IDF cosine) ─────────────────────────────────

def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """TF-IDF cosine similarity between two texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, min_df=1)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0])
    except Exception:
        return 0.0


def check_grounding_score(
    llm_output: Dict[str, Any],
    evidence_snippets: List[Dict[str, Any]],
    threshold: float = GROUNDING_THRESHOLD_DEFAULT,
) -> Tuple[bool, float, str]:
    """
    TF-IDF cosine between combined LLM output fields and combined evidence text.
    Returns (passes_threshold, score, message).
    """
    if not evidence_snippets:
        return False, 0.0, "GROUNDING_SCORE=0.000 (no evidence)"

    evidence_text = " ".join(c.get("chunk_text", "") for c in evidence_snippets)
    output_text = " ".join([
        llm_output.get("summary", ""),
        " ".join(llm_output.get("primary_indications", [])),
        llm_output.get("mechanism_summary", ""),
        llm_output.get("safety_notes", ""),
    ])

    score = compute_tfidf_similarity(output_text, evidence_text)
    passes = score >= threshold
    msg = f"GROUNDING_SCORE={score:.3f} (threshold={threshold})"
    return passes, score, msg


# ── CHECK 4: Coverage Check ───────────────────────────────────────────────────

def extract_key_terms(text: str, top_k: int = 10) -> set:
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "all", "each", "some", "such", "no", "not", "only", "same", "so",
        "than", "too", "very", "and", "but", "or", "if", "also", "now",
        "its", "it", "this", "that", "these", "those", "which", "who",
    }
    tokens = [t for t in tokenize(text) if t not in _STOPWORDS]
    tokens.sort(key=len, reverse=True)
    return set(tokens[:top_k])


def check_coverage(
    llm_output: Dict[str, Any],
    evidence_snippets: List[Dict[str, Any]],
    min_ratio: float = MIN_COVERAGE_RATIO,
) -> Tuple[bool, float, str]:
    """
    Check what fraction of key output terms appear in the evidence.
    Returns (passes, ratio, message).
    """
    if not evidence_snippets:
        return False, 0.0, "COVERAGE_RATIO=0.000 (no evidence)"

    output_text = " ".join([
        llm_output.get("summary", ""),
        " ".join(llm_output.get("primary_indications", [])),
        llm_output.get("mechanism_summary", ""),
        llm_output.get("safety_notes", ""),
    ])
    output_terms = extract_key_terms(output_text, top_k=15)

    if not output_terms:
        return False, 0.0, "COVERAGE_RATIO=0.000 (no key terms in output)"

    evidence_text = " ".join(c.get("chunk_text", "") for c in evidence_snippets)
    evidence_terms = set(tokenize(evidence_text))

    covered = output_terms & evidence_terms
    ratio = len(covered) / len(output_terms)
    passes = ratio >= min_ratio
    msg = f"COVERAGE_RATIO={ratio:.3f} ({len(covered)}/{len(output_terms)} terms covered)"
    return passes, ratio, msg


# ── CHECK 5: Retrieval sufficiency ────────────────────────────────────────────

def check_retrieval_sufficiency(
    evidence_snippets: List[Dict[str, Any]],
    min_snippets: int = 1,
) -> Tuple[bool, str]:
    """Block if retrieval returned no evidence for modes that require it."""
    if len(evidence_snippets) < min_snippets:
        return False, f"NO_RELEVANT_EVIDENCE (got {len(evidence_snippets)}, need {min_snippets})"
    return True, f"RETRIEVAL_OK ({len(evidence_snippets)} snippets)"


# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — BINARY DECISION
# ════════════════════════════════════════════════════════════════════════════

class GuardrailResult(BaseModel):
    decision: str               # "ALLOW" or "BLOCK"
    rule_tag: Optional[str]     # e.g., "JSON_INVALID", "LOW_GROUNDING"
    grounding_score: float
    coverage_ratio: float
    messages: List[str]
    validated_output: Optional[Dict[str, Any]]
    safe_refusal_message: Optional[str] = None


def run_guardrail_checks(
    llm_content: str,
    evidence_snippets: List[Dict[str, Any]],
    grounding_threshold: float = GROUNDING_THRESHOLD_DEFAULT,
    coverage_min_ratio: float = MIN_COVERAGE_RATIO,
    validate_grounding: bool = True,
    validate_coverage: bool = True,
) -> GuardrailResult:
    """
    Run all evidence-based guardrail checks. Returns binary ALLOW or BLOCK.

    All modes use this for final decision-making.
    validate_grounding / validate_coverage flags control enforcement
    (set False for baseline comparison where scores are computed but not enforced —
    NOTE: for final demo, both are True for all modes to enforce evidence-bounded answers).
    """
    messages = []

    # CHECK 1: JSON parse
    json_valid, json_error, parsed = check_json_parse(llm_content)
    if not json_valid:
        return GuardrailResult(
            decision="BLOCK", rule_tag="JSON_INVALID",
            grounding_score=0.0, coverage_ratio=0.0,
            messages=[json_error or "JSON parse failed"],
            validated_output=None,
            safe_refusal_message=SAFE_REFUSAL_DEFAULT,
        )
    messages.append("JSON_PARSE: OK")

    # CHECK 2: Schema validation
    schema_valid, schema_issues = check_schema(parsed)
    if not schema_valid:
        return GuardrailResult(
            decision="BLOCK", rule_tag="SCHEMA_INVALID",
            grounding_score=0.0, coverage_ratio=0.0,
            messages=schema_issues,
            validated_output=None,
            safe_refusal_message=SAFE_REFUSAL_DEFAULT,
        )
    messages.append("SCHEMA_CHECK: OK")

    # CHECK 3: Grounding score
    grounding_passes, grounding_score, grounding_msg = check_grounding_score(
        parsed, evidence_snippets, threshold=grounding_threshold
    )
    if validate_grounding and not grounding_passes:
        return GuardrailResult(
            decision="BLOCK", rule_tag="LOW_GROUNDING",
            grounding_score=grounding_score, coverage_ratio=0.0,
            messages=messages + [grounding_msg],
            validated_output=None,
            safe_refusal_message=SAFE_REFUSAL_DEFAULT,
        )
    messages.append(grounding_msg if validate_grounding else f"[not enforced] {grounding_msg}")

    # CHECK 4: Coverage
    coverage_passes, coverage_ratio, coverage_msg = check_coverage(
        parsed, evidence_snippets, min_ratio=coverage_min_ratio
    )
    if validate_coverage and not coverage_passes:
        return GuardrailResult(
            decision="BLOCK", rule_tag="LOW_COVERAGE",
            grounding_score=grounding_score, coverage_ratio=coverage_ratio,
            messages=messages + [coverage_msg],
            validated_output=None,
            safe_refusal_message=SAFE_REFUSAL_DEFAULT,
        )
    messages.append(coverage_msg if validate_coverage else f"[not enforced] {coverage_msg}")

    return GuardrailResult(
        decision="ALLOW", rule_tag=None,
        grounding_score=grounding_score, coverage_ratio=coverage_ratio,
        messages=messages,
        validated_output=parsed,
        safe_refusal_message=None,
    )


# ── Plain-text response renderer ──────────────────────────────────────────────

_TRIVIAL_VALUES = {"", "not specified", "n/a", "none", "unknown", "n/k"}


def _is_trivial(text: str) -> bool:
    return text.strip().lower() in _TRIVIAL_VALUES


def render_plain_text_response(llm_output: Dict[str, Any]) -> str:
    """
    Convert structured LLM output (DrugEnrichment schema) to a plain-text
    chatbot response. Used when decision == ALLOW.

    If the LLM populated `direct_answer`, render the focused factoid first
    and keep supporting context terse — the user asked a specific question
    and a wall of text is unfocused. Falls back to full rendering if
    `direct_answer` is missing or trivial.
    """
    parts: List[str] = []

    direct = (llm_output.get("direct_answer") or "").strip()
    has_direct = bool(direct) and not _is_trivial(direct)

    summary = (llm_output.get("summary") or "").strip()

    if has_direct:
        # Focused factoid mode
        parts.append(direct)
        # Add summary only if it adds genuinely new context (not a near-copy)
        if summary and not _is_trivial(summary) and summary.lower() not in direct.lower():
            parts.append("")
            parts.append(f"Context: {summary}")
        # Always show safety notes if present — relevant to all drug info
        safety = (llm_output.get("safety_notes") or "").strip()
        if safety and not _is_trivial(safety):
            parts.append("")
            parts.append(f"Safety information: {safety}")
    else:
        # Full rendering (no factoid focus detected by the model)
        if summary and not _is_trivial(summary):
            parts.append(summary)

        indications = [
            i.strip() for i in (llm_output.get("primary_indications") or [])
            if i and not _is_trivial(i)
        ]
        if indications:
            parts.append("")
            parts.append("Primary uses:")
            for ind in indications:
                parts.append(f"  • {ind}")

        mechanism = (llm_output.get("mechanism_summary") or "").strip()
        if mechanism and not _is_trivial(mechanism):
            parts.append("")
            parts.append(f"How it works: {mechanism}")

        safety = (llm_output.get("safety_notes") or "").strip()
        if safety and not _is_trivial(safety):
            parts.append("")
            parts.append(f"Safety information: {safety}")

    if not parts:
        return "Information retrieved but content is limited. Please consult authoritative drug references."

    parts.append("")
    parts.append(
        "Note: This information is for educational purposes only and is not medical advice. "
        "Always consult a healthcare professional."
    )
    return "\n".join(parts).strip()


# ── Blocked response helpers ──────────────────────────────────────────────────

BLOCKED_RESPONSE = {
    "summary": "Unable to provide verified information about this query.",
    "primary_indications": [],
    "mechanism_summary": "Information requires verification from authoritative sources.",
    "safety_notes": "Consult a healthcare professional for accurate drug information.",
    "tags": ["blocked", "insufficient_evidence"],
}


def format_blocked_response(rule_tag: str) -> Dict[str, Any]:
    """Return schema-compliant blocked response dict."""
    return BLOCKED_RESPONSE.copy()
