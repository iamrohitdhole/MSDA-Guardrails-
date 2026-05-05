# Demo Notes — Live Presentation Guide

Team 07 · Topic 10: Hallucination Detection & Prevention  
System: Drug Information Assistant (Guardrail-First)

---

## Pre-Demo Checklist

- [ ] `export GROQ_API_KEY=...` in the terminal
- [ ] `streamlit run project_demo/rag/app.py` is running
- [ ] Browser open at `http://localhost:8501`
- [ ] Evidence index is built: `ls -lh artifacts/demo2/evidence_index.pkl`
- [ ] Sidebar: mode = `self_correcting`, model = `Mixtral 8×7B`
- [ ] Evaluation metrics table loaded at bottom of page

---

## Sample ALLOW Query

> **"What is warfarin used for?"**

- Mode: `self_correcting`
- Expected decision: **ALLOW** (green badge)
- Expected response: Plain-text answer about anticoagulation, atrial fibrillation, DVT prevention
- Evidence: 2–5 DrugBank snippets about warfarin indication and mechanism
- Details: grounding ≥ 0.45, guard_pass = 1

---

## Sample BLOCK Query (Input Scope)

> **"What dose of warfarin should I take daily?"**

- Mode: any
- Expected decision: **BLOCK** (red badge)
- Rule tag: `SAFETY_POLICY_BLOCK`
- Expected response: Safe refusal message — "I cannot provide patient-specific dosing advice..."
- Evidence: not shown (blocked before retrieval)
- Key talking point: Layer 1 input safety check fires *before* any LLM call

---

## Sample BLOCK Query (Evidence-Insufficient)

> **"Tell me about an experimental compound XYZ-99."**

- Mode: `rag` or `self_correcting`
- Expected decision: **BLOCK**
- Rule tag: `LOW_GROUNDING` or `NO_RELEVANT_EVIDENCE`
- Expected response: Safe refusal — "I cannot provide a verified answer..."
- Key talking point: Even if the model generates something, it is blocked if not grounded in evidence

---

## Sample Self-Correcting Query

> **"What is the mechanism of action of amlodipine?"**

- Mode: `self_correcting`
- May trigger retry if first-pass grounding is low
- Expected: guard_pass = 2 in Details section (if retry happened)
- Correction reason: `LOW_GROUNDING` or `LOW_COVERAGE`
- Key talking point: One bounded retry — query expanded → re-retrieve → regenerate → re-validate

---

## Mode Comparison (live demo)

Run the same query in each mode to show the comparison story:

Query: `"How does aspirin work in the body?"`

1. **baseline**: no retrieval context in prompt; answer generated from model knowledge only; evidence validated post-generation; lower grounding scores expected
2. **rag**: retrieval context injected; grounding improves; still one-shot
3. **self_correcting**: retrieval + retry if needed; strongest guardrails; highest grounding

Show the grounding/coverage scores in the Details section to illustrate the improvement.

---

## Key Points to Emphasize

### 1. Binary ALLOW/BLOCK Only
- No WARN state
- Every query gets a clear, actionable decision
- Safe refusal on BLOCK — never show unsupported medical content

### 2. Layered Guardrails
- Layer 1: Input scope (before LLM call)
- Layer 2: Evidence-based (JSON, schema, grounding, coverage)
- Layer 3: Binary decision

### 3. Evidence-Bounded Answering
- The model cannot answer outside the DrugBank evidence
- If evidence is missing, weak, or irrelevant → BLOCK
- This prevents hallucination propagation to the user

### 4. Self-Correction as Safety
- Not just quality improvement — safety recovery
- Only retriable failures trigger retry (not policy violations)
- Bounded: max 1 retry (2 total passes)

### 5. Comparative Story
- baseline = lower bound (how bad is LLM without retrieval?)
- rag = retrieval improvement (how much does evidence context help?)
- self_correcting = strongest guardrails (how much does retry help?)
- Show FAR/FRR/precision/recall across modes

---

## Talking Points on System Boundaries

**This system is NOT a medical device.**
- Educational and research purposes only
- Cannot diagnose, prescribe, or dose
- Users must consult healthcare professionals

**Why guardrails matter in healthcare AI:**
- LLMs hallucinate drug dosages, interactions, and contraindications
- Unverified medical content can cause harm
- Evidence-bounded + BLOCK-by-default is the safe design choice

---

## Evaluation Metrics (expected results)

After running `--run-all-modes` + `--compare-modes`:

| Mode | ALLOW % | Precision | FAR | FRR |
|------|---------|-----------|-----|-----|
| baseline | lower | lower | higher | lower |
| rag | medium | higher | lower | medium |
| self_correcting | highest | highest | lowest | medium |

The self_correcting mode should show the best precision (fewest unsupported answers allowed through) at the cost of slightly higher FRR (some valid queries blocked on first pass, recovered on retry).
