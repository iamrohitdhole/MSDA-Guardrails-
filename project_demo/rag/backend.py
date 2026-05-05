"""
LLM Generation Backend — fully local via Ollama (OpenAI-compatible server).

Default backend: local_openai → Ollama at http://localhost:11434/v1
No hosted API or API keys required.

Model resolution order:
  1. Explicit llm_model_id argument  (e.g., from --model CLI flag)
  2. LOCAL_LLM_MODEL environment variable
  3. LLM_MODEL environment variable  (secondary alias)
  4. DEFAULT_MODEL config ("mistral")

Supported local models (via `ollama pull <name>`):
  mistral, llama3, llama3.1, gemma2, phi3, mixtral, llama3.2

Backends:
  "local_openai"  — Ollama / vLLM OpenAI-compatible server  [DEFAULT]
  "existing_llm"  — legacy Groq hosted API  (not used by default)
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

from project_demo.rag import MODEL_BACKEND, DEFAULT_MODEL

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
CALL_INTERVAL = 0.5  # seconds between calls

# ── Ollama local model choices (shown in Streamlit UI) ────────────────────────
# Ollama tag → human label.  Add/remove based on what you have pulled locally.
LOCAL_MODEL_CHOICES: List[tuple] = [
    ("mistral (recommended — 7B, fast)", "mistral"),
    ("llama3 (Meta LLaMA 3 8B)", "llama3"),
    ("llama3.1 (Meta LLaMA 3.1 8B)", "llama3.1"),
    ("gemma2 (Google Gemma 2 9B)", "gemma2"),
    ("phi3 (Microsoft Phi-3 Mini)", "phi3"),
    ("mixtral (Mixtral 8x7B MoE)", "mixtral"),
    ("llama3.2 (Meta LLaMA 3.2 3B)", "llama3.2"),
]

# Backwards-compatible aliases
HOSTED_MODEL_CHOICES = LOCAL_MODEL_CHOICES
HOSTED_BASELINE_MODEL_CHOICES = LOCAL_MODEL_CHOICES
COMPARISON_MODEL_CHOICES = LOCAL_MODEL_CHOICES


# ── Model resolution ──────────────────────────────────────────────────────────

def resolve_model(llm_model_id: Optional[str] = None) -> str:
    """
    Resolve the Ollama model tag using priority order:
    1. Explicit argument  (--model CLI)
    2. LOCAL_LLM_MODEL env var
    3. LLM_MODEL env var
    4. DEFAULT_MODEL config ("mistral")
    """
    if llm_model_id and str(llm_model_id).strip():
        return str(llm_model_id).strip()
    for env_var in ("LOCAL_LLM_MODEL", "LLM_MODEL"):
        val = os.getenv(env_var, "").strip()
        if val:
            return val
    return DEFAULT_MODEL


# Backwards-compatible aliases
def effective_groq_model(llm_model_id: Optional[str] = None) -> str:
    return resolve_model(llm_model_id)


def effective_local_model(llm_model_id: Optional[str] = None) -> str:
    return resolve_model(llm_model_id)


# ── Ollama client ─────────────────────────────────────────────────────────────

def get_local_openai_client() -> Optional[Any]:
    """Return an OpenAI client pointed at the local Ollama server."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error(
            "openai package not installed. Run: pip install openai\n"
            "  (This is the OpenAI SDK used to talk to the local Ollama server, "
            "not the OpenAI cloud service.)"
        )
        return None
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1").strip()
    api_key = os.getenv("LOCAL_LLM_API_KEY", "ollama")  # Ollama accepts any non-empty string
    return OpenAI(base_url=base_url, api_key=api_key)


def check_ollama_available() -> tuple[bool, str]:
    """
    Check whether the configured Ollama (or OpenAI-compatible) server is reachable.

    Honors LOCAL_LLM_BASE_URL exactly. Strips a trailing '/v1' if present so we
    can hit the native Ollama '/api/tags' endpoint, which is the cheapest probe.
    Works for local (127.0.0.1) and deployed Ollama hosts.
    """
    import urllib.request

    base = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2):
            return True, f"Ollama reachable at {base}"
    except Exception as e:
        return False, (
            f"Ollama not reachable at {base} ({e}).\n"
            "Local: start with `ollama serve` and pull a model: `ollama pull mistral`.\n"
            "Remote: set LOCAL_LLM_BASE_URL=http://<host>:11434/v1"
        )


def _strip_markdown_json_fences(content: str) -> str:
    import re
    content = content.strip()
    if content.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if m:
            content = m.group(1).strip()
    return content


# ── Query prompt builders ─────────────────────────────────────────────────────

_SCHEMA_BLOCK = """{
  "direct_answer": "string — 1-2 sentence DIRECT factoid answer to the user's specific question; empty string if the user did not ask a specific factoid",
  "summary": "string — 2-3 sentence drug summary",
  "primary_indications": ["string", "..."],
  "mechanism_summary": "string — mechanism of action in plain language",
  "safety_notes": "string — critical safety info and contraindications",
  "tags": ["string", "..."]
}"""


# Maps regex patterns in the user query to (canonical_field, label, fallback_synonyms).
# canonical_field matches the chunk's field_name in the evidence index.
# Order matters — most specific patterns first.
import re as _re

_FIELD_PATTERNS = [
    ("half_life",            r"\bhalf[\- ]life\b|\bt[\-_ ]?1?/?2\b",                     "half-life"),
    ("food_interactions",    r"\bfood\b|\bgrapefruit\b|\balcohol\b|\bmeal",              "food interactions"),
    ("drug_interactions",    r"\binteract|\bdrug[\- ]drug\b|\bcombined\s+with\b",        "drug interactions"),
    ("toxicity",             r"\bside\s+effects?|\badverse|\btoxic|\boverdos|\bharm",    "side effects / toxicity"),
    ("absorption",           r"\babsorpt|\babsorbed|\bbioavailab",                       "absorption"),
    ("metabolism",           r"\bmetaboli[sz]|\bcyp\d|\bcytochrome",                     "metabolism"),
    ("route_of_elimination", r"\beliminat|\bexcret|\bclearance",                         "elimination / excretion"),
    ("protein_binding",      r"\bprotein[\- ]bind|\bbound\s+to\s+protein",               "protein binding"),
    ("pharmacodynamics",     r"\bpharmacodyn",                                           "pharmacodynamics"),
    ("mechanism_of_action",  r"\bmechanism|\bmode\s+of\s+action|\bhow\s+does\s+\w+\s+work\b", "mechanism of action"),
    ("indication",           r"\bused\s+for\b|\bindicat|\btreats?\b\b|\btreatment\s+for\b", "indication / use"),
]
_FIELD_REGEXES = [(field, _re.compile(pat, _re.IGNORECASE), label) for field, pat, label in _FIELD_PATTERNS]


def detect_field_focus(query: str) -> Optional[tuple]:
    """
    Identify the field the user is asking about, if any.
    Returns (canonical_field_name, human_label) or None.
    Used by the prompt builder to instruct the model to lead with a focused
    answer drawn from chunks of that field type.
    """
    if not query:
        return None
    for field, regex, label in _FIELD_REGEXES:
        if regex.search(query):
            return (field, label)
    return None


def _build_query_prompt(
    query: str,
    evidence_snippets: List[Dict[str, Any]],
    baseline_no_context: bool = False,
    strict_grounding: bool = False,
    asked_field: Optional[tuple] = None,
) -> str:
    """
    Build the LLM prompt.

    `asked_field` is an optional (canonical_name, human_label) tuple from
    detect_field_focus(). When set, the prompt instructs the model to lead
    its answer with a focused factoid from chunks of that field type, using
    the `direct_answer` JSON field.
    """
    focus_block = ""
    if asked_field:
        field_canonical, field_label = asked_field
        focus_block = (
            f"\nFOCUS:\n"
            f"- The user is asking specifically about: {field_label}.\n"
            f'- Look in the Evidence below for chunks whose Field is "{field_canonical}".\n'
            f"- If such evidence exists, write a SHORT (1-2 sentence) factoid in `direct_answer`,\n"
            f"  quoting the specific value(s) from those chunks. Do NOT pad with general drug background.\n"
            f"- If no such evidence is present, set `direct_answer` to an empty string and answer normally.\n"
        )

    if baseline_no_context:
        return f"""You are answering a drug information query for an educational system.

Return ONLY a single valid JSON object matching this schema exactly (no markdown, no extra text):

{_SCHEMA_BLOCK}

RULES:
- summary, mechanism_summary, safety_notes are REQUIRED.
- direct_answer is OPTIONAL — leave as empty string unless answering a specific factoid.
- Write "Not specified" when uncertain — do NOT invent clinical details.
- Do NOT provide dosing instructions, prescribing advice, or patient-specific guidance.
{focus_block}
Query: {query}

JSON output only:""".strip()

    context_parts = [
        f"Drug: {s.get('drug_name', 'Unknown')}\n"
        f"Field: {s.get('field_name', 'N/A')}\n"
        f"Content: {s.get('chunk_text', '')}"
        for s in evidence_snippets
    ]
    context = "\n\n".join(context_parts) if context_parts else "(no evidence retrieved)"

    strict_block = ""
    if strict_grounding:
        strict_block = (
            "\nSTRICT MODE: Every clinical claim MUST appear in the Evidence below. "
            "Write 'Not specified' for anything not supported by the evidence.\n"
        )

    return f"""You are answering a drug information query using retrieved evidence from a curated DrugBank database.

Return ONLY a single valid JSON object matching this schema exactly (no markdown, no extra text):

{_SCHEMA_BLOCK}

RULES:
- summary, mechanism_summary, safety_notes are REQUIRED.
- direct_answer is OPTIONAL but PREFERRED for factoid questions; leave as empty string otherwise.
- Be concise. Do NOT pad with background the user did not ask for.
- Base every clinical claim ONLY on the Evidence section below.
- Do NOT hallucinate or add information absent from the evidence.
- Write "Not specified" when the evidence does not cover a detail.
- Do NOT provide dosing instructions, prescribing advice, or patient-specific guidance.
{focus_block}{strict_block}
Query: {query}

Evidence:
{context}

JSON output only:""".strip()


# ── Local OpenAI-compatible (Ollama) call ─────────────────────────────────────

def _call_local(client: Any, model: str, prompt: str) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(CALL_INTERVAL)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise medical information assistant. "
                            "Output ONLY valid JSON matching the requested schema. "
                            "No markdown fences, no explanations, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.05,
            )
            content = _strip_markdown_json_fences(resp.choices[0].message.content or "")
            return json.loads(content)
        except json.JSONDecodeError as e:
            raw = resp.choices[0].message.content if resp else ""
            logger.warning("JSON decode error (attempt %d/%d): %s | raw: %.200s", attempt, MAX_RETRIES, e, raw)
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 1.5)
            else:
                raise RuntimeError(
                    f"Local LLM returned non-JSON after {MAX_RETRIES} attempts. "
                    f"Raw output: {raw[:300]!r}"
                ) from e
        except Exception as e:
            logger.warning("Local LLM error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)
            else:
                raise RuntimeError(
                    f"Ollama call failed after {MAX_RETRIES} retries: {e}\n"
                    "Is Ollama running? Try: ollama serve"
                ) from e
    raise RuntimeError("Local LLM call failed unexpectedly")


# ── Primary generation function ───────────────────────────────────────────────

def generate_query_response(
    query: str,
    evidence_snippets: List[Dict[str, Any]],
    backend: Optional[str] = None,
    llm_model_id: Optional[str] = None,
    baseline_no_context: bool = False,
    strict_grounding: bool = False,
    asked_field: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Generate structured LLM response for a user drug-information query via Ollama.

    Args:
        query:               User's natural-language question.
        evidence_snippets:   Retrieved BM25 evidence chunks.
        backend:             "local_openai" (default) or "existing_llm" (legacy Groq).
        llm_model_id:        Ollama model tag (resolved via resolve_model()).
        baseline_no_context: True for baseline mode — evidence not injected into prompt.
        strict_grounding:    True for self-correcting second pass — stricter grounding prompt.

    Returns:
        dict matching DrugEnrichment schema (not yet validated by guardrail).
    """
    if backend is None:
        backend = MODEL_BACKEND

    prompt = _build_query_prompt(
        query, evidence_snippets,
        baseline_no_context=baseline_no_context,
        strict_grounding=strict_grounding,
        asked_field=asked_field,
    )

    if backend == "local_openai":
        client = get_local_openai_client()
        model = resolve_model(llm_model_id)
        if client is None:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        if not model:
            raise RuntimeError(
                "No model specified. Set LOCAL_LLM_MODEL env var or pass --model.\n"
                "Example: LOCAL_LLM_MODEL=mistral  or  --model mistral"
            )
        return _call_local(client, model, prompt)

    if backend == "existing_llm":
        # Legacy Groq path — kept for backwards compat but not the default
        return _call_groq_legacy(prompt, llm_model_id)

    raise ValueError(
        f"Unsupported backend: {backend!r}. Valid: 'local_openai', 'existing_llm'."
    )


# ── Legacy Groq path (kept for backwards compat, not default) ─────────────────

def _call_groq_legacy(prompt: str, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Legacy Groq call — only used when backend='existing_llm' is explicitly set."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq package not installed and existing_llm backend requested.")
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. The default backend is now local_openai (Ollama). "
            "To use Groq, set MODEL_BACKEND=existing_llm and GROQ_API_KEY."
        )
    client = Groq(api_key=api_key)
    model = resolve_model(model_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(CALL_INTERVAL)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Output ONLY valid JSON. No markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.05,
            )
            content = _strip_markdown_json_fences(resp.choices[0].message.content or "")
            return json.loads(content)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)
            else:
                raise RuntimeError(f"Groq call failed: {e}") from e
    raise RuntimeError("Groq call failed unexpectedly")


# Backwards-compatible alias
def get_groq_client():
    """Deprecated — use local_openai backend instead."""
    logger.warning("get_groq_client() called but default backend is now local_openai (Ollama).")
    return None


def call_groq_query(client: Any, prompt: str, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Backwards-compatible wrapper."""
    return _call_groq_legacy(prompt, model_id)


# ── Legacy drug-enrichment helpers (used by airflow DAGs) ────────────────────

def build_prompt(drug_record: Dict[str, Any]) -> str:
    """Build enrichment prompt for a DrugBank drug record (DAG use)."""
    name = drug_record.get("name", "Unknown drug")
    description = drug_record.get("description", "N/A")
    indication = drug_record.get("indication", "N/A")
    mechanism = drug_record.get("mechanism_of_action", "N/A")
    pharmacodynamics = drug_record.get("pharmacodynamics", "N/A")
    cas = drug_record.get("cas_number", "N/A")
    groups = ", ".join(drug_record.get("groups") or []) if isinstance(drug_record.get("groups"), list) else str(drug_record.get("groups", ""))
    synonyms = ", ".join(drug_record.get("synonyms") or []) if isinstance(drug_record.get("synonyms"), list) else str(drug_record.get("synonyms", ""))
    drug_id = drug_record.get("drugbank_id") or drug_record.get("id") or "unknown_id"

    return f"""You are helping to enrich DrugBank records.

Return ONLY valid JSON (no markdown):

{{
  "summary": "string",
  "primary_indications": ["string", ...],
  "mechanism_summary": "string",
  "safety_notes": "string",
  "tags": ["string", ...]
}}

Rules: all fields required, use "Not specified" when uncertain.

Drug: {name} ({drug_id})
Indication: {indication}
Mechanism: {mechanism}
Description: {description}

JSON only:""".strip()


def generate_llm_response(
    drug_record: Dict[str, Any],
    backend: Optional[str] = None,
    llm_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate LLM enrichment for a drug record (legacy DAG path)."""
    if backend is None:
        backend = MODEL_BACKEND
    prompt = build_prompt(drug_record)
    if backend == "local_openai":
        client = get_local_openai_client()
        model = resolve_model(llm_model_id)
        if not client or not model:
            raise RuntimeError("local_openai: missing client or model")
        return _call_local(client, model, prompt)
    if backend == "existing_llm":
        return _call_groq_legacy(prompt, llm_model_id)
    raise ValueError(f"Unsupported backend: {backend!r}")
