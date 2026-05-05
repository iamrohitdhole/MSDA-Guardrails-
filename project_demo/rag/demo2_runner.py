"""
End-to-end inference pipeline for the guardrail-first drug-information chatbot.

All three modes (baseline / rag / self_correcting) share:
  - Input scope / safety checks (Layer 1 guardrail)
  - Evidence-based guardrail checks (Layer 2)
  - Binary ALLOW / BLOCK decision (Layer 3)

Mode differences:
  baseline       — generates WITHOUT retrieval context; retrieves evidence POST-generation
                   for validation only; lower-bound comparison mode.
  rag            — retrieves BEFORE generation, injects evidence into prompt; comparison mode.
  self_correcting — rag + one bounded retry on retriable failures; default user-facing mode.

Final-demo safety rule: unsupported responses are BLOCKED in all modes.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from project_demo.rag import (
    GROUNDING_THRESHOLD_DEFAULT,
    MIN_COVERAGE_RATIO,
    SELF_CORRECT_MAX_PASSES,
    VALID_MODEL_MODES,
)
from project_demo.rag.backend import detect_field_focus, generate_query_response, resolve_model
from project_demo.rag.guardrail import (
    SAFE_REFUSAL_DEFAULT,
    GuardrailResult,
    check_input_scope,
    render_plain_text_response,
    run_guardrail_checks,
)
from project_demo.rag.retriever import EvidenceRetriever

# Rule tags that are safe to retry (not policy violations)
RETRY_RULE_TAGS = frozenset({"LOW_GROUNDING", "LOW_COVERAGE", "RETRIEVAL_FAILURE"})


def normalize_model_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m not in VALID_MODEL_MODES:
        raise ValueError(f"model_mode must be one of {VALID_MODEL_MODES!r}, got {mode!r}")
    return m


def mode_flags(model_mode: str) -> Tuple[bool, bool, bool, bool]:
    """
    Returns: (do_retrieve_before_gen, validate_grounding, validate_coverage, self_correct)

    baseline:       does NOT retrieve before generation (retrieves AFTER for post-validation).
                    For final demo, grounding/coverage are still validated using post-gen evidence.
    rag:            retrieves before generation; validates grounding + coverage.
    self_correcting: rag + one retry; validates grounding + coverage.
    """
    m = normalize_model_mode(model_mode)
    if m == "baseline":
        # do_retrieve_before=False; vg=True, vc=True enforce post-gen evidence validation
        return False, True, True, False
    if m == "rag":
        return True, True, True, False
    if m == "self_correcting":
        return True, True, True, True
    raise ValueError(m)


def expand_query_for_retry(query: str) -> str:
    """Widen recall for the second-pass retrieval in self_correcting mode."""
    tail = (
        "indication mechanism toxicity contraindications interactions "
        "pharmacodynamics adverse effects drug interactions safety"
    )
    return f"{query.strip()} {tail}".strip()


def _make_scope_blocked_record(
    prompt: Dict[str, Any],
    mode: str,
    model_id: Optional[str],
    scope_rule_tag: str,
    refusal_message: str,
    start: float,
) -> Dict[str, Any]:
    """Build a BLOCK record for an input scope failure."""
    latency_ms = (time.time() - start) * 1000
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": prompt.get("query", ""),
        "mode": mode,
        "model": model_id,
        "decision": "BLOCK",
        "final_text_response": refusal_message,
        "answer_text_internal": "",
        "summary": "",
        "primary_indications": [],
        "mechanism_summary": "",
        "safety_notes": "",
        "interaction_notes": "",
        "tags": [],
        "confidence_score": 0.0,
        "grounding_score": 0.0,
        "coverage_score": 0.0,
        "hallucination_risk": "high",
        "rule_tags": [scope_rule_tag],
        "citations": [],
        "evidence": [],
        "latency_ms": latency_ms,
        "safe_refusal_message": refusal_message,
        "refusal_reason": scope_rule_tag,
        # Legacy fields for evaluator compat
        "prompt_id": prompt.get("prompt_id"),
        "drug_id": prompt.get("drug_id"),
        "drug_name": prompt.get("drug_name"),
        "query": prompt.get("query", ""),
        "query_type": prompt.get("query_type"),
        "model_mode": mode,
        "guard_pass": 0,
        "first_pass_rule_tag": None,
        "evidence_count": 0,
        "retrieval_query_used": prompt.get("query", ""),
        "llm_content": "{}",
        "llm_output": {},
        "rule_tag": scope_rule_tag,
        "coverage_ratio": 0.0,
        "guardrail_messages": [f"INPUT_SCOPE_BLOCKED: {scope_rule_tag}"],
        "llm_model": model_id,
    }


def _build_output_record(
    prompt: Dict[str, Any],
    mode: str,
    model_id: Optional[str],
    decision: str,
    result: GuardrailResult,
    llm_output: Dict[str, Any],
    llm_content: str,
    evidence: List[Dict[str, Any]],
    guard_pass: int,
    first_pass_rule_tag: Optional[str],
    retrieval_query: str,
    start: float,
    # self_correcting extras
    initial_answer: Optional[Dict[str, Any]] = None,
    corrected_answer: Optional[Dict[str, Any]] = None,
    correction_applied: bool = False,
    correction_reason: Optional[str] = None,
    first_pass_rule_tags: Optional[List[str]] = None,
    second_pass_rule_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    latency_ms = (time.time() - start) * 1000
    validated = result.validated_output or {}

    if decision == "ALLOW":
        final_text = render_plain_text_response(validated)
        safe_refusal = None
        refusal_reason = None
        hallucination_risk = "low"
    else:
        final_text = result.safe_refusal_message or SAFE_REFUSAL_DEFAULT
        safe_refusal = final_text
        refusal_reason = result.rule_tag
        hallucination_risk = "blocked"

    citations = [
        {
            "evidence_id": f"{s.get('drug_id', 'unk')}_{s.get('field_name', 'unk')}_{s.get('chunk_idx', 0)}",
            "drug_id": s.get("drug_id"),
            "drug_name": s.get("drug_name"),
            "field_name": s.get("field_name"),
            "chunk_text": s.get("chunk_text", "")[:300],
            "retrieval_score": s.get("bm25_score", 0.0),
        }
        for s in evidence
    ]

    rec = {
        # Primary contract fields (spec §10)
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": prompt.get("query", ""),
        "mode": mode,
        "model": model_id,
        "decision": decision,
        "final_text_response": final_text,
        "answer_text_internal": json.dumps(llm_output),
        "summary": validated.get("summary", ""),
        "primary_indications": validated.get("primary_indications", []),
        "mechanism_summary": validated.get("mechanism_summary", ""),
        "safety_notes": validated.get("safety_notes", ""),
        "interaction_notes": "",
        "tags": validated.get("tags", []),
        "confidence_score": result.grounding_score,
        "grounding_score": result.grounding_score,
        "coverage_score": result.coverage_ratio,
        "hallucination_risk": hallucination_risk,
        "rule_tags": [result.rule_tag] if result.rule_tag else [],
        "citations": citations,
        "evidence": evidence,
        "latency_ms": latency_ms,
        # BLOCK fields
        "safe_refusal_message": safe_refusal,
        "refusal_reason": refusal_reason,
    }

    # Self-correcting extras
    if mode == "self_correcting":
        rec.update({
            "initial_answer": initial_answer or {},
            "corrected_answer": corrected_answer or {},
            "correction_applied": correction_applied,
            "correction_reason": correction_reason,
            "first_pass_rule_tags": first_pass_rule_tags or [],
            "second_pass_rule_tags": second_pass_rule_tags or [],
        })

    # Legacy / evaluator compatibility fields
    rec.update({
        "prompt_id": prompt.get("prompt_id"),
        "drug_id": prompt.get("drug_id"),
        "drug_name": prompt.get("drug_name"),
        "query": prompt.get("query", ""),
        "query_type": prompt.get("query_type"),
        "model_mode": mode,
        "guard_pass": guard_pass,
        "first_pass_rule_tag": first_pass_rule_tag,
        "evidence_count": len(evidence),
        "retrieval_query_used": retrieval_query,
        "llm_content": llm_content,
        "llm_output": llm_output,
        "rule_tag": result.rule_tag,
        "coverage_ratio": result.coverage_ratio,
        "guardrail_messages": result.messages,
        "llm_model": model_id,
    })
    return rec


def run_demo2_single(
    prompt: Dict[str, Any],
    retriever: Optional[EvidenceRetriever],
    backend: str = "local_openai",
    top_k: int = 5,
    llm_model_id: Optional[str] = None,
    model_mode: str = "self_correcting",
    grounding_threshold: float = GROUNDING_THRESHOLD_DEFAULT,
    coverage_min_ratio: float = MIN_COVERAGE_RATIO,
    self_correct_max_passes: int = SELF_CORRECT_MAX_PASSES,
) -> Dict[str, Any]:
    """
    Run one request through the full guardrail pipeline.

    Layer 1 — Input scope check (shared, all modes).
    Layer 2 — Mode-specific generation + evidence-based guardrail.
    Layer 3 — Binary ALLOW / BLOCK with plain-text rendering.
    """
    start = time.time()
    mode = normalize_model_mode(model_mode)
    model_id = resolve_model(llm_model_id)
    query = str(prompt.get("query", "")).strip()

    do_retrieve_before, vg, vc, self_correct = mode_flags(mode)
    asked_field = detect_field_focus(query)  # → (canonical, label) or None

    # ── Layer 1: Input scope check ────────────────────────────────────────────
    scope = check_input_scope(query)
    if scope.blocked:
        return _make_scope_blocked_record(
            prompt, mode, model_id,
            scope.rule_tag or "SAFETY_POLICY_BLOCK",
            scope.refusal_message or SAFE_REFUSAL_DEFAULT,
            start,
        )

    evidence: List[Dict[str, Any]] = []
    llm_output: Dict[str, Any] = {}
    llm_content: str = "{}"
    guard_pass = 1
    first_pass_rule_tag_on_retry: Optional[str] = None
    initial_answer: Optional[Dict[str, Any]] = None
    corrected_answer: Optional[Dict[str, Any]] = None
    correction_applied = False
    correction_reason: Optional[str] = None
    first_pass_rule_tags: List[str] = []
    second_pass_rule_tags: List[str] = []
    retrieval_query = query

    # ── Layer 2: Generation + guardrail ──────────────────────────────────────

    if mode == "baseline":
        # Generate WITHOUT retrieval context (lower-bound comparison)
        llm_output = generate_query_response(
            query, [],
            backend=backend, llm_model_id=model_id,
            baseline_no_context=True, strict_grounding=False,
            asked_field=asked_field,
        )
        llm_content = json.dumps(llm_output)

        # Post-generation evidence retrieval for validation
        if retriever is not None:
            evidence = retriever.retrieve_evidence(query, top_k=top_k)

        result = run_guardrail_checks(
            llm_content, evidence,
            grounding_threshold=grounding_threshold,
            coverage_min_ratio=coverage_min_ratio,
            validate_grounding=vg,
            validate_coverage=vc,
        )

    elif mode == "rag":
        if retriever is None:
            raise ValueError("EvidenceRetriever is required for rag mode.")
        evidence = retriever.retrieve_evidence(query, top_k=top_k)
        llm_output = generate_query_response(
            query, evidence,
            backend=backend, llm_model_id=model_id,
            baseline_no_context=False, strict_grounding=False,
            asked_field=asked_field,
        )
        llm_content = json.dumps(llm_output)
        result = run_guardrail_checks(
            llm_content, evidence,
            grounding_threshold=grounding_threshold,
            coverage_min_ratio=coverage_min_ratio,
            validate_grounding=vg,
            validate_coverage=vc,
        )

    elif mode == "self_correcting":
        if retriever is None:
            raise ValueError("EvidenceRetriever is required for self_correcting mode.")

        # First pass
        evidence = retriever.retrieve_evidence(query, top_k=top_k)
        llm_output = generate_query_response(
            query, evidence,
            backend=backend, llm_model_id=model_id,
            baseline_no_context=False, strict_grounding=False,
            asked_field=asked_field,
        )
        llm_content = json.dumps(llm_output)
        result = run_guardrail_checks(
            llm_content, evidence,
            grounding_threshold=grounding_threshold,
            coverage_min_ratio=coverage_min_ratio,
            validate_grounding=True,
            validate_coverage=True,
        )
        first_pass_rule_tags = [result.rule_tag] if result.rule_tag else []
        initial_answer = dict(llm_output)

        # One retry only (max_passes = 2 = first pass + one correction)
        max_passes = min(int(self_correct_max_passes), SELF_CORRECT_MAX_PASSES)
        if (
            guard_pass < max_passes
            and result.decision == "BLOCK"
            and result.rule_tag in RETRY_RULE_TAGS
        ):
            first_pass_rule_tag_on_retry = result.rule_tag
            correction_reason = result.rule_tag

            q_retry = expand_query_for_retry(query)
            retrieval_query = q_retry
            evidence_retry = retriever.retrieve_evidence(q_retry, top_k=top_k)

            llm_output = generate_query_response(
                query, evidence_retry,
                backend=backend, llm_model_id=model_id,
                baseline_no_context=False, strict_grounding=True,
                asked_field=asked_field,
            )
            llm_content = json.dumps(llm_output)
            result = run_guardrail_checks(
                llm_content, evidence_retry,
                grounding_threshold=grounding_threshold,
                coverage_min_ratio=coverage_min_ratio,
                validate_grounding=True,
                validate_coverage=True,
            )
            evidence = evidence_retry
            guard_pass = 2
            correction_applied = True
            corrected_answer = dict(llm_output)
            second_pass_rule_tags = [result.rule_tag] if result.rule_tag else []

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # ── Layer 3: Build output record ──────────────────────────────────────────
    return _build_output_record(
        prompt=prompt,
        mode=mode,
        model_id=model_id,
        decision=result.decision,
        result=result,
        llm_output=llm_output,
        llm_content=llm_content,
        evidence=evidence,
        guard_pass=guard_pass,
        first_pass_rule_tag=first_pass_rule_tag_on_retry,
        retrieval_query=retrieval_query,
        start=start,
        initial_answer=initial_answer,
        corrected_answer=corrected_answer,
        correction_applied=correction_applied,
        correction_reason=correction_reason,
        first_pass_rule_tags=first_pass_rule_tags,
        second_pass_rule_tags=second_pass_rule_tags,
    )
