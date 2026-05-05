"""
Drug Information Assistant — main chatbot page.

Final-demo build:
  - Single fixed LLM (resolve_model() — defaults to mistral; engineering env
    vars LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL still apply for deployment).
  - Single fixed mode: self_correcting (not surfaced in the UI).
  - Frozen guardrail thresholds (not surfaced in the UI).
  - Chat-style interface. No mode badges, no rule-tag labels, no evaluation
    proxy fields, no debug controls. The system decides ALLOW vs BLOCK
    internally and renders the appropriate plain-text reply.

Other Streamlit pages (Mode Comparison, Model Comparison) live under pages/
and are auto-discovered by Streamlit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from project_demo.rag import (
    GROUNDING_THRESHOLD_DEFAULT,
    MIN_COVERAGE_RATIO,
    SELF_CORRECT_MAX_PASSES,
)
from project_demo.rag.backend import check_ollama_available, resolve_model
from project_demo.rag.demo2_runner import run_demo2_single
from project_demo.storage.artifact_store import get_evidence_index, status_summary

# ── Frozen demo configuration ────────────────────────────────────────────────
FIXED_MODE = "self_correcting"
FIXED_TOP_K = 5
GREETING = (
    "Hi — ask me about a drug's uses, side effects, mechanism of action, "
    "interactions, or pharmacokinetics."
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Information Assistant",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="auto",
)


# ── Cached retriever (loaded once per session) ───────────────────────────────
@st.cache_resource
def _load_retriever():
    info = get_evidence_index()
    if info.origin == "missing":
        return None
    from project_demo.rag.retriever import EvidenceRetriever
    r = EvidenceRetriever()
    r.load_index(info.path)
    return r


# ── Sidebar — system status only (no controls) ───────────────────────────────
with st.sidebar:
    # Streamlit's auto page-nav renders above this block.
    st.markdown("### System")
    ok_llm, _ = check_ollama_available()
    st.markdown(
        f"{'🟢' if ok_llm else '🔴'} Language model"
        f"\n\n{'🟢' if status_summary()['reachable'] else '🟡'} Evidence store"
    )
    retriever = _load_retriever()
    if retriever is None:
        st.markdown("🔴 Evidence index")
    else:
        st.markdown("🟢 Evidence index")


# ── Main chat ────────────────────────────────────────────────────────────────
st.title("Drug Information Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show greeting only when there's no history yet
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(GREETING)

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about a drug…")
if prompt:
    user_text = prompt.strip()
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking…_")

            if retriever is None or not ok_llm:
                # Surface a friendly message without engineering jargon
                reply = (
                    "I'm not able to answer right now — the system is starting up. "
                    "Please try again in a moment."
                )
            else:
                try:
                    out = run_demo2_single(
                        {
                            "prompt_id": 0, "drug_id": "", "drug_name": "",
                            "query": user_text, "query_type": "interactive",
                        },
                        retriever,
                        backend="local_openai",
                        top_k=FIXED_TOP_K,
                        llm_model_id=resolve_model(),
                        model_mode=FIXED_MODE,
                        grounding_threshold=float(GROUNDING_THRESHOLD_DEFAULT),
                        coverage_min_ratio=float(MIN_COVERAGE_RATIO),
                        self_correct_max_passes=SELF_CORRECT_MAX_PASSES,
                    )
                    reply = (
                        out.get("final_text_response")
                        or "I couldn't find a verified answer to that question."
                    )
                except Exception:
                    reply = (
                        "Something went wrong while answering. "
                        "Please try a different question."
                    )

            placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Small footer (kept minimal — single line) ────────────────────────────────
st.caption("For educational use. Not medical advice.")
