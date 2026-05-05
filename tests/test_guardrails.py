"""
Tests for the guardrail-first drug-information chatbot.

Covers:
- Binary ALLOW/BLOCK only (no WARN state)
- Input scope checks (diagnosis, prescribing, patient-specific dosing)
- Evidence-based guardrail checks (grounding, coverage, JSON, schema)
- Self-correcting retry limit = 1
- Plain-text response rendering
- Internal output schema
- Mode-specific behavior
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_demo.rag import SELF_CORRECT_MAX_PASSES, VALID_MODEL_MODES
from project_demo.rag.guardrail import (
    SAFE_REFUSAL_DEFAULT,
    GuardrailResult,
    check_input_scope,
    check_json_parse,
    check_schema,
    check_grounding_score,
    check_coverage,
    render_plain_text_response,
    run_guardrail_checks,
)
from project_demo.rag.demo2_runner import (
    RETRY_RULE_TAGS,
    expand_query_for_retry,
    mode_flags,
    normalize_model_mode,
    run_demo2_single,
)
from project_demo.rag.backend import resolve_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_LLM_OUTPUT = {
    "summary": "Aspirin is a non-steroidal anti-inflammatory drug used for pain relief and fever reduction.",
    "primary_indications": ["pain relief", "fever reduction", "cardiovascular prophylaxis"],
    "mechanism_summary": "Inhibits cyclooxygenase enzymes COX-1 and COX-2, reducing prostaglandin synthesis.",
    "safety_notes": "Risk of gastrointestinal bleeding; avoid in children with viral illness due to Reye syndrome.",
    "tags": ["analgesic", "anti-inflammatory", "antiplatelet", "nsaid"],
}

VALID_EVIDENCE = [
    {
        "drug_id": "DB00945",
        "drug_name": "Aspirin",
        "field_name": "mechanism_of_action",
        "chunk_idx": 0,
        "chunk_text": (
            "Aspirin inhibits cyclooxygenase enzymes COX-1 and COX-2 thereby blocking "
            "prostaglandin and thromboxane synthesis. This results in anti-inflammatory, "
            "analgesic, antipyretic, and antiplatelet effects."
        ),
        "bm25_score": 0.82,
    },
    {
        "drug_id": "DB00945",
        "drug_name": "Aspirin",
        "field_name": "indication",
        "chunk_idx": 0,
        "chunk_text": (
            "Aspirin is indicated for the treatment of mild to moderate pain, fever, "
            "and inflammation. It is also used for cardiovascular prophylaxis to prevent "
            "myocardial infarction and stroke."
        ),
        "bm25_score": 0.71,
    },
]

WEAK_EVIDENCE = [
    {
        "drug_id": "DB99999",
        "drug_name": "SomeDrug",
        "field_name": "description",
        "chunk_idx": 0,
        "chunk_text": "A compound used in various applications.",
        "bm25_score": 0.1,
    }
]


# ════════════════════════════════════════════════════════════════════════════
# 1. BINARY ALLOW/BLOCK — NO WARN STATE
# ════════════════════════════════════════════════════════════════════════════

class TestBinaryDecision:
    def test_decision_is_allow_or_block(self):
        result = run_guardrail_checks(
            json.dumps(VALID_LLM_OUTPUT), VALID_EVIDENCE,
            grounding_threshold=0.1, coverage_min_ratio=0.1,
        )
        assert result.decision in ("ALLOW", "BLOCK")

    def test_no_warn_state_in_valid_modes(self):
        assert "WARN" not in VALID_MODEL_MODES

    def test_allow_with_good_evidence(self):
        result = run_guardrail_checks(
            json.dumps(VALID_LLM_OUTPUT), VALID_EVIDENCE,
            grounding_threshold=0.1, coverage_min_ratio=0.1,
        )
        assert result.decision == "ALLOW"

    def test_block_with_no_evidence(self):
        result = run_guardrail_checks(
            json.dumps(VALID_LLM_OUTPUT), [],
            grounding_threshold=0.1, coverage_min_ratio=0.1,
        )
        assert result.decision == "BLOCK"

    def test_block_with_low_grounding(self):
        result = run_guardrail_checks(
            json.dumps(VALID_LLM_OUTPUT), WEAK_EVIDENCE,
            grounding_threshold=0.9,  # Very high threshold
            coverage_min_ratio=0.0,
            validate_grounding=True,
        )
        assert result.decision == "BLOCK"
        assert result.rule_tag == "LOW_GROUNDING"

    def test_block_with_invalid_json(self):
        result = run_guardrail_checks("not valid json {{{", VALID_EVIDENCE)
        assert result.decision == "BLOCK"
        assert result.rule_tag == "JSON_INVALID"

    def test_block_with_invalid_schema(self):
        bad_output = {"summary": "Test"}  # Missing required fields
        result = run_guardrail_checks(
            json.dumps(bad_output), VALID_EVIDENCE,
            grounding_threshold=0.0, coverage_min_ratio=0.0,
        )
        assert result.decision == "BLOCK"
        assert result.rule_tag == "SCHEMA_INVALID"


# ════════════════════════════════════════════════════════════════════════════
# 2. INPUT SCOPE CHECKS
# ════════════════════════════════════════════════════════════════════════════

class TestInputScope:
    def test_safe_drug_query_passes(self):
        result = check_input_scope("What is warfarin used for?")
        assert not result.blocked

    def test_mechanism_query_passes(self):
        result = check_input_scope("How does aspirin work in the body?")
        assert not result.blocked

    def test_side_effects_query_passes(self):
        result = check_input_scope("What are the side effects of metformin?")
        assert not result.blocked

    def test_diagnosis_request_blocked(self):
        result = check_input_scope("Can you diagnose my condition?")
        assert result.blocked
        # Blocked at Layer 1 (OUT_OF_SCOPE_QUERY) because no drug keywords,
        # or at Layer 2 if phrased with drug keywords
        assert result.rule_tag in ("OUT_OF_SCOPE_QUERY", "OUT_OF_SCOPE_MEDICAL_ADVICE")

    def test_do_i_have_blocked(self):
        result = check_input_scope("Do I have diabetes based on my symptoms?")
        assert result.blocked

    def test_prescribe_me_blocked(self):
        result = check_input_scope("Can you prescribe me metformin?")
        assert result.blocked
        assert result.rule_tag == "SAFETY_POLICY_BLOCK"

    def test_patient_specific_dosing_blocked(self):
        result = check_input_scope("How much warfarin should I take daily?")
        assert result.blocked
        assert result.rule_tag == "SAFETY_POLICY_BLOCK"

    def test_my_dosage_blocked(self):
        result = check_input_scope("What is my dosage for aspirin?")
        assert result.blocked

    def test_treat_my_blocked(self):
        result = check_input_scope("Treat my high blood pressure with medication.")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_MEDICAL_ADVICE"

    def test_overdose_blocked(self):
        result = check_input_scope("What happens if I overdose on acetaminophen?")
        assert result.blocked
        assert result.rule_tag == "SAFETY_POLICY_BLOCK"

    def test_blocked_has_refusal_message(self):
        result = check_input_scope("Prescribe me warfarin.")
        assert result.blocked
        assert result.refusal_message is not None
        assert len(result.refusal_message) > 10

    # ── Off-topic query tests (Layer 1: drug-relatedness check) ─────────────────

    def test_off_topic_greeting_how_are_you_blocked(self):
        result = check_input_scope("How are you?")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"
        assert "drug-related" in result.refusal_message.lower() or "scope" in result.refusal_message.lower()

    def test_off_topic_what_is_your_name_blocked(self):
        result = check_input_scope("What is your name?")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"

    def test_off_topic_tell_me_joke_blocked(self):
        result = check_input_scope("Tell me a joke.")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"

    def test_off_topic_weather_blocked(self):
        result = check_input_scope("What's the weather like today?")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"

    def test_off_topic_recipe_blocked(self):
        result = check_input_scope("How do I make a chocolate cake?")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"

    def test_off_topic_math_problem_blocked(self):
        result = check_input_scope("What is 2 + 2?")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"

    def test_off_topic_sports_blocked(self):
        result = check_input_scope("Who won the Super Bowl?")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"

    def test_off_topic_general_health_no_drug_blocked(self):
        result = check_input_scope("How do I stay healthy?")
        assert result.blocked
        assert result.rule_tag == "OUT_OF_SCOPE_QUERY"

    def test_drug_query_with_keyword_passes(self):
        result = check_input_scope("What is aspirin?")
        assert not result.blocked

    def test_drug_query_with_medication_keyword_passes(self):
        result = check_input_scope("Tell me about metformin medication.")
        assert not result.blocked

    def test_drug_query_with_generic_name_passes(self):
        result = check_input_scope("What is acetylsalicylic acid used for?")
        assert not result.blocked

    def test_drug_query_with_pharma_term_passes(self):
        result = check_input_scope("Explain the pharmacokinetics of ibuprofen.")
        assert not result.blocked

    def test_drug_query_with_interaction_keyword_passes(self):
        result = check_input_scope("What drug interactions does warfarin have?")
        assert not result.blocked

    def test_drug_query_with_mechanism_keyword_passes(self):
        result = check_input_scope("What is the mechanism of action of sertraline?")
        assert not result.blocked

    def test_drug_query_with_side_effect_keyword_passes(self):
        result = check_input_scope("What are adverse effects of statins?")
        assert not result.blocked

    def test_drug_query_with_contraindication_keyword_passes(self):
        result = check_input_scope("What are contraindications for NSAIDs?")
        assert not result.blocked

    def test_drug_query_with_dosage_keyword_passes_scope_but_blocks_later(self):
        """Queries with 'dosage' keyword pass drug-relatedness check but may fail other scope checks."""
        result = check_input_scope("What is the dosage of warfarin?")
        # This should pass drug-relatedness but may be blocked on patient-specific dosing check
        # Just verify it gets past Layer 1 drug-relatedness check
        if result.blocked:
            # If blocked, it should be on a different rule (patient-specific dosing)
            assert result.rule_tag != "OUT_OF_SCOPE_QUERY"
        # If not blocked by Layer 2, it passes scope checks
        else:
            assert not result.blocked


# ════════════════════════════════════════════════════════════════════════════
# 3. EVIDENCE-INSUFFICIENT → BLOCK IN ALL MODES
# ════════════════════════════════════════════════════════════════════════════

class TestEvidenceInsufficient:
    def _make_prompt(self, query: str = "What is aspirin?") -> Dict[str, Any]:
        return {"prompt_id": 0, "drug_id": "DB00945", "drug_name": "Aspirin",
                "query": query, "query_type": "test"}

    def _mock_retriever(self, evidence: List[Dict]) -> MagicMock:
        r = MagicMock()
        r.retrieve_evidence.return_value = evidence
        return r

    def _mock_llm(self, output: Dict) -> str:
        return "project_demo.rag.backend.generate_query_response"

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_rag_blocks_when_no_evidence(self, mock_gen):
        mock_gen.return_value = VALID_LLM_OUTPUT
        retriever = self._mock_retriever([])

        out = run_demo2_single(
            self._make_prompt(), retriever,
            backend="existing_llm", model_mode="rag",
            grounding_threshold=0.1, coverage_min_ratio=0.1,
        )
        assert out["decision"] == "BLOCK"

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_self_correcting_blocks_when_no_evidence(self, mock_gen):
        mock_gen.return_value = VALID_LLM_OUTPUT
        retriever = self._mock_retriever([])

        out = run_demo2_single(
            self._make_prompt(), retriever,
            backend="existing_llm", model_mode="self_correcting",
            grounding_threshold=0.1, coverage_min_ratio=0.1,
        )
        assert out["decision"] == "BLOCK"

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_baseline_blocks_when_no_evidence(self, mock_gen):
        mock_gen.return_value = VALID_LLM_OUTPUT
        retriever = self._mock_retriever([])

        out = run_demo2_single(
            self._make_prompt(), retriever,
            backend="existing_llm", model_mode="baseline",
            grounding_threshold=0.1, coverage_min_ratio=0.1,
        )
        assert out["decision"] == "BLOCK"


# ════════════════════════════════════════════════════════════════════════════
# 4. SELF-CORRECTING: RETRY LIMIT = 1
# ════════════════════════════════════════════════════════════════════════════

class TestSelfCorrectingRetry:
    def test_max_passes_is_two(self):
        """SELF_CORRECT_MAX_PASSES must equal 2 (first pass + one correction)."""
        assert SELF_CORRECT_MAX_PASSES == 2

    def test_retry_rule_tags(self):
        """Only retriable failures trigger a retry."""
        assert "LOW_GROUNDING" in RETRY_RULE_TAGS
        assert "LOW_COVERAGE" in RETRY_RULE_TAGS
        assert "SCHEMA_INVALID" not in RETRY_RULE_TAGS
        assert "SAFETY_POLICY_BLOCK" not in RETRY_RULE_TAGS
        assert "OUT_OF_SCOPE_MEDICAL_ADVICE" not in RETRY_RULE_TAGS

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_at_most_one_retry(self, mock_gen):
        """Self-correcting makes at most 2 LLM calls (first pass + one retry)."""
        # First call returns something that will fail grounding; second call also fails
        mock_gen.return_value = VALID_LLM_OUTPUT

        retriever = MagicMock()
        retriever.retrieve_evidence.return_value = []  # No evidence → LOW_GROUNDING

        out = run_demo2_single(
            {"prompt_id": 0, "drug_id": "", "drug_name": "", "query": "How does aspirin work?", "query_type": "test"},
            retriever,
            backend="existing_llm",
            model_mode="self_correcting",
            grounding_threshold=0.99,
            coverage_min_ratio=0.99,
            self_correct_max_passes=2,
        )
        # Should have made at most 2 LLM calls
        assert mock_gen.call_count <= 2
        assert out["decision"] in ("ALLOW", "BLOCK")

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_policy_block_no_retry(self, mock_gen):
        """Scope policy blocks must NOT trigger a retry."""
        out = run_demo2_single(
            {"prompt_id": 0, "drug_id": "", "drug_name": "",
             "query": "Prescribe me warfarin.", "query_type": "test"},
            MagicMock(),
            backend="existing_llm",
            model_mode="self_correcting",
        )
        # Input scope check should fire before LLM is called
        assert mock_gen.call_count == 0
        assert out["decision"] == "BLOCK"
        assert out.get("rule_tag") in ("SAFETY_POLICY_BLOCK", "OUT_OF_SCOPE_MEDICAL_ADVICE")

    def test_expand_query_for_retry(self):
        q = "What is warfarin?"
        expanded = expand_query_for_retry(q)
        assert q in expanded
        assert "mechanism" in expanded
        assert "indication" in expanded


# ════════════════════════════════════════════════════════════════════════════
# 5. PLAIN-TEXT RENDERING
# ════════════════════════════════════════════════════════════════════════════

class TestPlainTextRendering:
    def test_render_produces_string(self):
        text = render_plain_text_response(VALID_LLM_OUTPUT)
        assert isinstance(text, str)
        assert len(text) > 20

    def test_render_contains_summary(self):
        text = render_plain_text_response(VALID_LLM_OUTPUT)
        # Summary should appear (or at least the key content)
        assert "aspirin" in text.lower() or "anti-inflammatory" in text.lower()

    def test_render_contains_disclaimer(self):
        text = render_plain_text_response(VALID_LLM_OUTPUT)
        assert "educational" in text.lower() or "healthcare" in text.lower()

    def test_render_no_raw_json_keys(self):
        text = render_plain_text_response(VALID_LLM_OUTPUT)
        assert '"summary"' not in text
        assert '"primary_indications"' not in text
        assert '"mechanism_summary"' not in text

    def test_render_not_specified_filtered(self):
        sparse_output = {
            "summary": "Test drug summary.",
            "primary_indications": ["Not specified"],
            "mechanism_summary": "Not specified",
            "safety_notes": "Some safety info.",
            "tags": ["test"],
        }
        text = render_plain_text_response(sparse_output)
        # "Not specified" entries should be filtered out or handled gracefully
        assert "Test drug summary." in text


# ════════════════════════════════════════════════════════════════════════════
# 6. INTERNAL OUTPUT SCHEMA
# ════════════════════════════════════════════════════════════════════════════

class TestOutputSchema:
    REQUIRED_KEYS = [
        "id", "timestamp", "question", "mode", "model", "decision",
        "final_text_response", "grounding_score", "coverage_score",
        "rule_tags", "citations", "evidence", "latency_ms",
    ]
    SELF_CORRECTING_KEYS = [
        "correction_applied", "correction_reason",
        "first_pass_rule_tags", "second_pass_rule_tags",
    ]

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_output_has_required_fields(self, mock_gen):
        mock_gen.return_value = VALID_LLM_OUTPUT
        retriever = MagicMock()
        retriever.retrieve_evidence.return_value = VALID_EVIDENCE

        out = run_demo2_single(
            {"prompt_id": 0, "drug_id": "DB00945", "drug_name": "Aspirin",
             "query": "What is aspirin?", "query_type": "test"},
            retriever,
            backend="existing_llm", model_mode="rag",
            grounding_threshold=0.0, coverage_min_ratio=0.0,
        )
        for key in self.REQUIRED_KEYS:
            assert key in out, f"Missing required field: {key}"

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_self_correcting_output_has_correction_fields(self, mock_gen):
        mock_gen.return_value = VALID_LLM_OUTPUT
        retriever = MagicMock()
        retriever.retrieve_evidence.return_value = VALID_EVIDENCE

        out = run_demo2_single(
            {"prompt_id": 0, "drug_id": "DB00945", "drug_name": "Aspirin",
             "query": "What is aspirin?", "query_type": "test"},
            retriever,
            backend="existing_llm", model_mode="self_correcting",
            grounding_threshold=0.0, coverage_min_ratio=0.0,
        )
        for key in self.SELF_CORRECTING_KEYS:
            assert key in out, f"Missing self_correcting field: {key}"

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_block_has_safe_refusal(self, mock_gen):
        mock_gen.return_value = VALID_LLM_OUTPUT
        retriever = MagicMock()
        retriever.retrieve_evidence.return_value = []

        out = run_demo2_single(
            {"prompt_id": 0, "drug_id": "", "drug_name": "",
             "query": "What is aspirin?", "query_type": "test"},
            retriever,
            backend="existing_llm", model_mode="rag",
            grounding_threshold=0.1, coverage_min_ratio=0.1,
        )
        if out["decision"] == "BLOCK":
            assert out.get("safe_refusal_message") is not None
            assert len(out["safe_refusal_message"]) > 10


# ════════════════════════════════════════════════════════════════════════════
# 7. MODE-SPECIFIC BEHAVIOR
# ════════════════════════════════════════════════════════════════════════════

class TestModeFlags:
    def test_baseline_no_retrieve_before_gen(self):
        do_retrieve, vg, vc, sc = mode_flags("baseline")
        assert not do_retrieve, "baseline must NOT retrieve before generation"
        assert not sc, "baseline must NOT self-correct"

    def test_baseline_validates_evidence(self):
        do_retrieve, vg, vc, sc = mode_flags("baseline")
        # For final demo safety, baseline enforces grounding/coverage on post-gen evidence
        assert vg, "baseline must validate grounding (post-gen)"
        assert vc, "baseline must validate coverage (post-gen)"

    def test_rag_retrieves_before_gen(self):
        do_retrieve, vg, vc, sc = mode_flags("rag")
        assert do_retrieve, "rag must retrieve before generation"
        assert not sc, "rag must NOT self-correct"
        assert vg and vc, "rag must validate grounding and coverage"

    def test_self_correcting_retrieves_and_retries(self):
        do_retrieve, vg, vc, sc = mode_flags("self_correcting")
        assert do_retrieve
        assert vg and vc
        assert sc, "self_correcting must have self_correct=True"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            normalize_model_mode("warn")
        with pytest.raises(ValueError):
            normalize_model_mode("unknown_mode")

    @patch("project_demo.rag.demo2_runner.generate_query_response")
    def test_baseline_calls_llm_without_context(self, mock_gen):
        mock_gen.return_value = VALID_LLM_OUTPUT
        retriever = MagicMock()
        retriever.retrieve_evidence.return_value = VALID_EVIDENCE

        run_demo2_single(
            {"prompt_id": 0, "query": "What is aspirin?", "query_type": "test"},
            retriever,
            backend="existing_llm", model_mode="baseline",
            grounding_threshold=0.0, coverage_min_ratio=0.0,
        )
        # baseline_no_context must be True for baseline
        call_kwargs = mock_gen.call_args
        assert call_kwargs is not None
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        if not kwargs:
            args = call_kwargs.args
            # positional: query, evidence_snippets, backend, llm_model_id, baseline_no_context
            # baseline_no_context is at position 4
            if len(args) > 4:
                assert args[4] is True
        else:
            assert kwargs.get("baseline_no_context", True) is True


# ════════════════════════════════════════════════════════════════════════════
# 8. MODEL RESOLUTION
# ════════════════════════════════════════════════════════════════════════════

class TestModelResolution:
    def test_explicit_arg_wins(self):
        model = resolve_model("my-explicit-model")
        assert model == "my-explicit-model"

    def test_env_local_llm_model(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_MODEL", "local-model-tag")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        model = resolve_model(None)
        assert model == "local-model-tag"

    def test_env_llm_model_fallback(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        monkeypatch.setenv("LLM_MODEL", "llm-fallback")
        model = resolve_model(None)
        assert model == "llm-fallback"

    def test_config_default_fallback(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        from project_demo.rag import DEFAULT_MODEL
        model = resolve_model(None)
        assert model == DEFAULT_MODEL


# ════════════════════════════════════════════════════════════════════════════
# 9. JSON / SCHEMA CHECKS
# ════════════════════════════════════════════════════════════════════════════

class TestJsonSchemaChecks:
    def test_valid_json_parse(self):
        valid, err, parsed = check_json_parse(json.dumps(VALID_LLM_OUTPUT))
        assert valid
        assert err is None
        assert parsed == VALID_LLM_OUTPUT

    def test_json_with_markdown_fence(self):
        fenced = "```json\n" + json.dumps(VALID_LLM_OUTPUT) + "\n```"
        valid, err, parsed = check_json_parse(fenced)
        assert valid

    def test_invalid_json(self):
        valid, err, parsed = check_json_parse("{bad json}")
        assert not valid
        assert "JSON_PARSE_ERROR" in (err or "")

    def test_valid_schema(self):
        valid, issues = check_schema(VALID_LLM_OUTPUT)
        assert valid
        assert issues == []

    def test_invalid_schema_missing_field(self):
        bad = {"summary": "Test"}
        valid, issues = check_schema(bad)
        assert not valid
        assert len(issues) > 0


# ════════════════════════════════════════════════════════════════════════════
# 10. GROUNDING AND COVERAGE
# ════════════════════════════════════════════════════════════════════════════

class TestGroundingCoverage:
    def test_grounding_score_with_relevant_evidence(self):
        passes, score, msg = check_grounding_score(VALID_LLM_OUTPUT, VALID_EVIDENCE, threshold=0.1)
        assert passes
        assert score > 0.0

    def test_grounding_score_no_evidence(self):
        passes, score, msg = check_grounding_score(VALID_LLM_OUTPUT, [], threshold=0.1)
        assert not passes
        assert score == 0.0

    def test_coverage_with_relevant_evidence(self):
        passes, ratio, msg = check_coverage(VALID_LLM_OUTPUT, VALID_EVIDENCE, min_ratio=0.1)
        assert passes
        assert ratio > 0.0

    def test_coverage_no_evidence(self):
        passes, ratio, msg = check_coverage(VALID_LLM_OUTPUT, [], min_ratio=0.1)
        assert not passes
        assert ratio == 0.0
