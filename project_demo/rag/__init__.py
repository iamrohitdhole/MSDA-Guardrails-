"""
Guardrail-first drug-information chatbot — Demo 2 (Topic 10: Hallucination Detection & Prevention).

Architecture: curated DrugBank evidence → BM25 retrieval → LLM generation → layered guardrails → ALLOW/BLOCK.

LLM backend: local Ollama (OpenAI-compatible server at http://localhost:11434/v1).
No hosted API or API keys required. Runs fully locally with open-weight models.

Three modes:
  baseline       — direct generation; post-generation evidence validation; lower-bound comparison.
  rag            — retrieval + generation + grounding/coverage guardrails; retrieval comparison mode.
  self_correcting — retrieval + generation + full guardrails + one bounded retry; default user-facing mode.

Final-demo safety rule: unsupported responses (not grounded in retrieved evidence) are BLOCKED in all modes.
"""

import os
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "demo2"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Artifact paths ────────────────────────────────────────────────────────────
EVIDENCE_INDEX_PATH = ARTIFACTS_DIR / "evidence_index.pkl"
INFERENCE_OUTPUTS_PATH = ARTIFACTS_DIR / "inference_outputs.jsonl"
# Legacy alias (typo-safe)
INFECTION_OUTPUTS_PATH = INFERENCE_OUTPUTS_PATH
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# ── LLM backend ───────────────────────────────────────────────────────────────
# "local_openai" → Ollama at http://localhost:11434/v1  (fully local, no API key needed)
# "existing_llm" → legacy Groq hosted API path (not used by default)
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "local_openai")

# ── Model resolution (CLI > LOCAL_LLM_MODEL env > LLM_MODEL env > default) ───
# Default: mistral — widely available in Ollama, good instruction-following, runs on CPU
# Pull with: ollama pull mistral
DEFAULT_MODEL = "mistral"
# Resolved at runtime by resolve_model() in backend.py

# ── Mode config ───────────────────────────────────────────────────────────────
# self_correcting is the default normal-user mode for the final demo
MODEL_MODE = os.getenv("MODEL_MODE", "self_correcting")
VALID_MODEL_MODES = ("baseline", "rag", "self_correcting")

# One retry = 2 total passes (first pass + one correction pass)
SELF_CORRECT_MAX_PASSES = 2

# ── Guardrail thresholds ──────────────────────────────────────────────────────
# These are FROZEN for the final demo. Validated end-to-end with mocked + real
# Mistral runs in Milestones 2.5–2.6:
#   - grounded answers land at grounding ≥ 0.50, coverage ≥ 0.85
#   - ungrounded answers land at grounding ≤ 0.22, coverage ≤ 0.05
# The 0.45 / 0.30 cuts give a safe ALLOW/BLOCK margin without false-positives
# on either side. Do NOT expose these as user-tunable sliders in the demo UI.
GROUNDING_THRESHOLD_DEFAULT = 0.45
MIN_COVERAGE_RATIO = 0.30
THRESHOLDS_FROZEN = True  # informational marker for the UI

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 50

# Fields to chunk from drug records.
#
# These MUST be raw DrugBank-derived fields. No LLM-generated text (summary,
# safety_notes, etc.) is allowed in the retrieval/grounding basis — that is a
# project-level invariant; see project_demo/pipeline/build_evidence.py.
#
# `normalize_drug_record` in retriever.py maps legacy aliases onto these
# canonical keys for backward compatibility with the older Silver schema.
CHUNK_FIELDS = [
    "description",
    "indication",
    "mechanism_of_action",
    "pharmacodynamics",
    "toxicity",
    "absorption",
    "half_life",
    "metabolism",
    "protein_binding",
    "route_of_elimination",
    "drug_interactions",
    "food_interactions",
]

# ── BM25 config ───────────────────────────────────────────────────────────────
BM25_K1 = 1.5
BM25_B = 0.75

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_EVERY = 10
