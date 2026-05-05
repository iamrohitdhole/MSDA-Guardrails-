# Guardrails Implementation Summary

## Overview
This document summarizes the guardrail-first safety system implementation for the drug-information chatbot (Topic 10: Hallucination Detection & Prevention).

## Architecture

### Three-Layer Guardrail Design

#### **Layer 1: Input Scope Check**
Validates whether the query is drug-related before any LLM inference.

- **Drug-Relatedness Check**: Uses regex pattern with 50+ drug/pharmaceutical keywords
  - Keywords include drug names (aspirin, warfarin, metformin, etc.)
  - Keywords include drug classes (antacid, antibiotic, statin, NSAID, etc.)
  - Keywords include medical terms (side effects, mechanism, interactions, contraindications, etc.)
  - Handles plurals: "side effects", "contraindications", "statins", "NSAIDs"

- **Non-Drug Queries**: Immediately blocked with `OUT_OF_SCOPE_QUERY` rule tag
  - Examples: "How are you?", "What's the weather?", "Tell me a joke"
  - User message: "I can only answer drug-related questions..."

#### **Layer 2: Safety Policy Checks**
For drug-related queries, validates against unsafe request patterns.

- **Diagnosis Requests** → Blocked with `OUT_OF_SCOPE_MEDICAL_ADVICE`
  - Pattern: "diagnose", "do i have", "is it cancer", etc.
  - Examples: "Can you diagnose me?", "Do I have diabetes?"

- **Prescribing Requests** → Blocked with `SAFETY_POLICY_BLOCK`
  - Pattern: "prescribe me", "write me a prescription", etc.
  - Examples: "Prescribe me warfarin", "Can you prescribe aspirin?"

- **Patient-Specific Dosing** → Blocked with `SAFETY_POLICY_BLOCK`
  - Pattern: "how much [drug] should i take", "my dosage", "dose for my age", etc.
  - Examples: "How much warfarin should I take?", "What dose for my weight?"

- **Treatment Planning** → Blocked with `OUT_OF_SCOPE_MEDICAL_ADVICE`
  - Pattern: "treat my", "switch my medication", etc.
  - Examples: "Treat my high blood pressure", "Is it safe for me to take?"

- **Overdose/Self-Harm** → Blocked with `SAFETY_POLICY_BLOCK`
  - Pattern: "overdose", "lethal dose", "suicide", etc.
  - Examples: "What's the lethal dose?", "How to overdose?"

#### **Layer 3: Evidence-Based Checks**
For queries passing Layers 1-2, validates LLM output against retrieved evidence.

- **JSON Parse**: Validates output is valid JSON
- **Schema Validation**: Verifies against DrugEnrichment schema (summary, indications, mechanism, safety_notes, tags)
- **Grounding Score**: TF-IDF cosine similarity between LLM output and retrieved evidence
- **Coverage Ratio**: Fraction of output key terms present in evidence

### Binary Decision Model
- **ALLOW**: Query passes all scope checks, LLM output parses and validates, meets grounding/coverage thresholds
- **BLOCK**: Any scope check fails, JSON/schema validation fails, grounding/coverage below thresholds
- **No WARN state**: All decisions are binary

## Implementation Files

### Core Guardrail Module
**`project_demo/rag/guardrail.py`**
- `check_input_scope(query)`: Layer 1 & 2 checks
- `run_guardrail_checks(llm_content, evidence)`: Layer 3 checks
- `GuardrailResult`: Output model with decision, rule_tag, messages, scores

### Supporting Modules
- **`retriever.py`**: BM25 evidence retrieval with markdown cleanup
- **`backend.py`**: Ollama local inference client
- **`demo2_runner.py`**: Mode-specific inference orchestration
- **`app.py`**: Streamlit interactive UI

## Testing

### Comprehensive Test Suite
**`tests/test_guardrails.py`**: 70 tests covering:

1. **Binary Decision Tests** (7 tests)
   - Verifies ALLOW/BLOCK decisions, no WARN state

2. **Input Scope Tests** (28 tests)
   - Off-topic queries blocked (17 new tests)
   - Valid drug queries pass (11 existing tests)
   - All safety policy blocks verified

3. **Evidence Sufficiency Tests** (3 tests)
   - All modes block with no evidence

4. **Self-Correcting Tests** (5 tests)
   - Max 2 passes (1 retry), retry rule tags

5. **Rendering & Schema Tests** (10 tests)
   - Plain-text response rendering
   - Output schema validation

6. **Mode & Resolution Tests** (10 tests)
   - Mode-specific behavior
   - Model resolution order

7. **Grounding & Coverage Tests** (7 tests)
   - TF-IDF similarity scoring
   - Key term coverage validation

### Test Results
✓ All 70 tests passing
✓ Off-topic queries correctly blocked
✓ Valid drug queries pass scope check
✓ Safety policy blocks working
✓ Evidence-based checks functioning

## Key Behaviors

### Off-Topic Query Blocking
```python
# Examples - all blocked at Layer 1 with OUT_OF_SCOPE_QUERY
queries = [
    "How are you?",              # No drug keywords
    "What's the weather?",       # No drug keywords
    "Tell me a joke",            # No drug keywords
    "Who won the Super Bowl?",   # No drug keywords
]

# Examples - pass Layer 1, allowed if evidence sufficient
queries = [
    "What is aspirin?",          # Has "aspirin" keyword
    "How does warfarin work?",   # Has "warfarin" keyword
    "What are side effects of ibuprofen?",  # Has "side effects" + "ibuprofen"
]
```

### Safety Policy Blocking
```python
# Patient-specific dosing - blocked at Layer 2
"How much warfarin should I take?"  # Blocked: SAFETY_POLICY_BLOCK

# Prescribing - blocked at Layer 2
"Can you prescribe metformin?"      # Blocked: SAFETY_POLICY_BLOCK

# Overdose - blocked at Layer 2
"What's the lethal dose?"           # Blocked: SAFETY_POLICY_BLOCK
```

### Evidence-Based Allowing
```python
# Valid query with good evidence - allowed
Query: "What is aspirin used for?"
Evidence: [BM25 snippets from aspirin mechanism, indications, etc.]
LLM Output: Valid JSON matching DrugEnrichment schema
Grounding Score: 0.72 (above 0.45 threshold)
Coverage Ratio: 0.85 (above 0.60 threshold)
Decision: ALLOW

# Valid query but weak evidence - blocked
Query: "What is aspirin used for?"
Evidence: [Unrelated drug mentions, weak matches]
Grounding Score: 0.15 (below 0.45 threshold)
Decision: BLOCK (LOW_GROUNDING)
```

## Configuration

### Environment Variables
- `LOCAL_LLM_MODEL`: Override model selection (e.g., "llama3" instead of "mistral")
- `LLM_MODEL`: Fallback model selection

### Model Selection Priority
1. Explicit CLI argument `--model`
2. `LOCAL_LLM_MODEL` environment variable
3. `LLM_MODEL` environment variable
4. `DEFAULT_MODEL` in config (mistral)

### Thresholds
- `GROUNDING_THRESHOLD_DEFAULT`: 0.45 (TF-IDF cosine similarity)
- `MIN_COVERAGE_RATIO`: 0.60 (fraction of output terms in evidence)
- `SELF_CORRECT_MAX_PASSES`: 2 (first pass + one retry)

## Usage

### Running Tests
```bash
# Run all guardrail tests
python3 -m pytest tests/test_guardrails.py -v

# Run specific test class
python3 -m pytest tests/test_guardrails.py::TestInputScope -v

# Run off-topic query tests only
python3 -m pytest tests/test_guardrails.py::TestInputScope::test_off_topic_* -v
```

### Running the Interactive Demo
```bash
# Start Ollama (in separate terminal)
ollama serve

# Pull a model (if not already pulled)
ollama pull mistral

# Run Streamlit app
streamlit run project_demo/rag/app.py
```

### Building Evidence Index
```bash
python -m project_demo.rag.build_index --from-parquet silver/silver_drugs_from_delta.parquet
```

## Design Decisions

1. **Binary ALLOW/BLOCK Only**: No WARN state simplifies decision logic and forces conservative decisions
2. **Drug-Relatedness as Layer 1**: Blocks off-topic conversations immediately, before expensive LLM inference
3. **Regex-Based Pattern Matching**: Fast, interpretable, no ML dependencies for scope checks
4. **TF-IDF for Grounding**: Simple baseline for evidence relevance; captures token overlap
5. **Self-Correction (1 retry max)**: Allows recovery from weak first pass but prevents infinite loops
6. **Local Inference via Ollama**: No cloud dependencies, full privacy, complete offline operation

## Future Improvements

1. **Semantic Retrieval**: Replace BM25 with dense embeddings for better drug name matching
2. **Drug Synonym Expansion**: Handle aspirin vs. acetylsalicylic acid aliasing
3. **Contextual Safety Scores**: ML-based safety classification instead of regex patterns
4. **Multilingual Support**: Extend drug keywords to other languages
5. **User Feedback Loop**: Collect corrections from real usage to refine thresholds
