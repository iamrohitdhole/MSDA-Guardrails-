# Drug Information Assistant — Guardrail-First Safety System

**Team 07 · MSDA Final Demo · Topic 10: Hallucination Detection & Prevention**

A healthcare-domain chatbot that answers drug-information questions with strict evidence-bounded guardrails. Every response is grounded in curated DrugBank data and carries a binary **ALLOW** or **BLOCK** decision.

---

## System Architecture

```
Curated DrugBank evidence
  └── Cleaning / normalization (silver pipeline)
      └── Chunking (150–200 tokens) + BM25 indexing
          └── EvidenceRetriever (artifacts/demo2/evidence_index.pkl)

Runtime:
  User Query
    → [Layer 1] Input Safety / Scope Checks
        • Block diagnosis requests
        • Block prescribing advice
        • Block patient-specific dosing
        • Block unsafe personalized medical advice
    → Mode Router (baseline | rag | self_correcting)
    → [Layer 2] Evidence-based Guardrails
        • JSON parse validity
        • Schema validation
        • Grounding score (TF-IDF cosine vs evidence)
        • Coverage check (key terms in evidence)
    → [Layer 3] Decision Engine: ALLOW or BLOCK
    → Response Renderer (plain text)
    → Streamlit UI
```

---

## Three Modes

| Mode | Purpose | Retrieval | Self-correct |
|------|---------|-----------|-------------|
| `baseline` | Lower-bound comparison | No context in prompt; post-gen validation | No |
| `rag` | Retrieval comparison | Retrieved evidence injected into prompt | No |
| `self_correcting` | **Default user-facing mode** | Retrieved evidence + one bounded retry | Yes (1 retry) |

### Safety Rule (all modes)
If the LLM response is not grounded in retrieved DrugBank evidence, it is **BLOCKED** in all modes. The model is never allowed to answer from general world knowledge when evidence is absent or insufficient.

### Mode Details

**baseline** — generates without retrieval context (lower-bound comparison). Evidence is retrieved *after* generation for post-generation validation. If the answer fails grounding/coverage checks, it is blocked.

**rag** — retrieves top-k evidence snippets and injects them into the generation prompt. Validates grounding and coverage. Shows what retrieval context improves over baseline.

**self_correcting** — the default safe mode. Retrieval + generation + full guardrails + one bounded retry. On a retriable failure (LOW_GROUNDING, LOW_COVERAGE), the query is expanded, evidence is re-retrieved, and the model regenerates under a stricter grounding prompt. Maximum one retry (two passes total).

---

## Guardrail Design

### Input Scope Checks (Layer 1)
Applied to all modes before any LLM call:

| Pattern | Rule Tag |
|---------|----------|
| Diagnosis requests ("do I have", "diagnose me") | `OUT_OF_SCOPE_MEDICAL_ADVICE` |
| Prescribing advice ("prescribe me", "write me a prescription") | `SAFETY_POLICY_BLOCK` |
| Patient-specific dosing ("how much should I take", "my dosage") | `SAFETY_POLICY_BLOCK` |
| Treatment planning ("treat my", "cure my") | `OUT_OF_SCOPE_MEDICAL_ADVICE` |
| Overdose / self-harm | `SAFETY_POLICY_BLOCK` |

### Evidence-Based Checks (Layer 2)

| Check | Rule Tag on Failure |
|-------|-------------------|
| JSON parse | `JSON_INVALID` |
| Schema validation | `SCHEMA_INVALID` |
| Grounding score (TF-IDF cosine) | `LOW_GROUNDING` |
| Term coverage | `LOW_COVERAGE` |
| No evidence retrieved | `NO_RELEVANT_EVIDENCE` |

### Decision (Layer 3)
- **ALLOW**: passes all checks → plain-text response shown to user
- **BLOCK**: any check fails → safe refusal message shown; no medical content exposed

---

## Local Models (Ollama)

All inference runs fully locally via [Ollama](https://ollama.ai). No API key or internet access required at runtime.

| Ollama tag | Notes |
|------------|-------|
| `mistral` | **Recommended** — 7B, fast, good instruction-following |
| `llama3` | Meta LLaMA 3 8B |
| `llama3.1` | Meta LLaMA 3.1 8B |
| `gemma2` | Google Gemma 2 9B |
| `phi3` | Microsoft Phi-3 Mini — lightweight |
| `mixtral` | Mixtral 8x7B MoE — larger, slower |
| `llama3.2` | Meta LLaMA 3.2 3B — fastest, smallest |

Pull a model before use: `ollama pull mistral`

Model resolution order: `--model` CLI → `LOCAL_LLM_MODEL` env → `LLM_MODEL` env → config default (`mistral`)

---

## System Scope

**This system IS:**
- Educational
- Evidence-grounded (DrugBank-derived)
- Guardrail-first

**This system is NOT:**
- A diagnosis tool
- A treatment recommendation engine
- A prescribing system
- A patient-specific dosing assistant
- A regulated medical device

---

## Quick Start

See [RUNBOOK.md](RUNBOOK.md) for detailed setup, index building, inference, evaluation, and demo commands.

```bash
# 1. Install dependencies and start Ollama
pip install openai streamlit scikit-learn pandas numpy pydantic
ollama serve &          # start local inference server
ollama pull mistral     # pull the default model

# 2. Build evidence index
python -m project_demo.rag.build_index --from-parquet silver/silver_drugs_from_delta.parquet

# 3. Launch chatbot
streamlit run project_demo/rag/app.py

# 4. Run batch inference (all modes)
python -m project_demo.rag.run_inference --run-all-modes --model mistral

# 5. Run evaluation
python -m project_demo.rag.run_evaluation --compare-modes

# 6. Run tests
python -m pytest tests/ -v
```

---

## Repository Structure

```
project_demo/
├── rag/
│   ├── __init__.py          # Config, constants, paths
│   ├── app.py               # Streamlit chatbot UI (main demo)
│   ├── backend.py           # LLM generation (Ollama via local OpenAI-compatible server)
│   ├── guardrail.py         # All guardrail layers (scope + evidence + decision)
│   ├── retriever.py         # BM25 evidence indexing and retrieval
│   ├── demo2_runner.py      # End-to-end pipeline (all three modes)
│   ├── build_index.py       # CLI: build evidence index
│   ├── run_inference.py     # CLI: batch inference
│   ├── run_evaluation.py    # CLI: evaluation metrics + plots
│   └── evaluator.py         # Metrics computation and plot generation
├── eda/
│   └── app_streamlit.py     # Legacy EDA dashboard
silver/
└── silver_drugs_from_delta.parquet/   # 11,524 DrugBank records
artifacts/
└── demo2/
    ├── evidence_index.pkl              # Built BM25 index
    ├── inference_baseline.jsonl        # Baseline inference outputs
    ├── inference_rag.jsonl             # RAG inference outputs
    ├── inference_self_correcting.jsonl # Self-correcting inference outputs
    ├── metrics.json                    # Single-mode metrics
    ├── metrics_by_mode.json            # Cross-mode comparison
    └── plots/                          # ROC, confusion matrix, latency plots
tests/
└── test_guardrails.py       # 53 unit tests
```
