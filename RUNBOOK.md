# Runbook — Drug Information Assistant

## Prerequisites

**1. Install Python dependencies**
```bash
pip install openai streamlit scikit-learn pandas numpy pydantic
```

**2. Install and start Ollama**

macOS (Homebrew):
```bash
brew install ollama
ollama serve          # starts the local inference server at http://localhost:11434
```

Or download from [ollama.ai](https://ollama.ai) and run the app.

**3. Pull a model**
```bash
ollama pull mistral   # recommended default — 7B, fast, good instruction-following
```

Other supported models:
```bash
ollama pull llama3
ollama pull llama3.1
ollama pull gemma2
ollama pull phi3
ollama pull llama3.2
```

**4. Verify Ollama is running**
```bash
curl http://localhost:11434/api/tags
```

You should see a JSON list of your pulled models. If Ollama is not running, start it with `ollama serve`.

---

## 1. Build the Evidence Index

Build from the local silver parquet (no Spark required):

```bash
python -m project_demo.rag.build_index \
    --from-parquet silver/silver_drugs_from_delta.parquet
```

The index is saved to `artifacts/demo2/evidence_index.pkl`.

With a sample limit (for quick testing):
```bash
python -m project_demo.rag.build_index \
    --from-parquet silver/silver_drugs_from_delta.parquet \
    --sample-limit 500
```

Verify the index was built:
```bash
ls -lh artifacts/demo2/evidence_index.pkl
```

---

## 2. Run Inference

### Single mode
```bash
# Self-correcting (default / recommended)
python -m project_demo.rag.run_inference \
    --mode self_correcting \
    --model mistral \
    --n-prompts 100

# RAG (retrieval comparison)
python -m project_demo.rag.run_inference \
    --mode rag \
    --model mistral \
    --n-prompts 100

# Baseline (lower bound)
python -m project_demo.rag.run_inference \
    --mode baseline \
    --model mistral \
    --n-prompts 100
```

### All three modes (same prompts, for comparison)
```bash
python -m project_demo.rag.run_inference \
    --run-all-modes \
    --model mistral \
    --n-prompts 100
```

### With explicit paths
```bash
python -m project_demo.rag.run_inference \
    --mode self_correcting \
    --model mistral \
    --input artifacts/demo2/evidence_index.pkl \
    --output artifacts/demo2/inference_self_correcting.jsonl \
    --n-prompts 200
```

### Model alternatives
```bash
# mistral (recommended — 7B, fast)
--model mistral

# LLaMA 3 8B
--model llama3

# LLaMA 3.1 8B
--model llama3.1

# Google Gemma 2 9B
--model gemma2

# Microsoft Phi-3 Mini
--model phi3

# Mixtral 8x7B (larger, slower)
--model mixtral

# LLaMA 3.2 3B (lightweight)
--model llama3.2
```

### Model resolution
The model is resolved in this order:
1. `--model` CLI argument
2. `LOCAL_LLM_MODEL` environment variable
3. `LLM_MODEL` environment variable (secondary alias)
4. Config default: `mistral`

Example using env var instead of CLI flag:
```bash
export LOCAL_LLM_MODEL=gemma2
python -m project_demo.rag.run_inference --mode self_correcting --n-prompts 50
```

---

## 3. Run Evaluation

### Compare all three modes
```bash
python -m project_demo.rag.run_evaluation --compare-modes
```

Outputs:
- `artifacts/demo2/metrics_by_mode.json` — cross-mode metrics table
- `artifacts/demo2/plots/latency_by_mode.png` — latency comparison bar chart

### Single-mode evaluation
```bash
python -m project_demo.rag.run_evaluation \
    --inference-path artifacts/demo2/inference_self_correcting.jsonl \
    --metrics-path artifacts/demo2/metrics.json
```

Outputs:
- `artifacts/demo2/metrics.json`
- `artifacts/demo2/plots/roc_curve.png`
- `artifacts/demo2/plots/confusion_matrix.png`
- `artifacts/demo2/plots/grounding_distribution.png`
- `artifacts/demo2/plots/latency_distribution.png`

---

## 4. Launch the Streamlit Chatbot

```bash
streamlit run project_demo/rag/app.py
```

The app opens at `http://localhost:8501`.

The sidebar shows the Ollama connection status (green = reachable, red = not running). Default mode is `self_correcting`. Switch modes in the sidebar for comparison.

---

## 5. Run Tests

```bash
python -m pytest tests/ -v
```

Expected: 53 tests pass.

---

## Recommended Live Demo Flow

1. **Set up** — start Ollama and the app:
   ```bash
   ollama serve &
   streamlit run project_demo/rag/app.py
   ```

2. **ALLOW demo** — show a successful drug query:
   - Query: `"What is warfarin used for?"`
   - Mode: `self_correcting`
   - Expected: green ALLOW badge, plain-text response, evidence snippets

3. **BLOCK demo (scope)** — show an out-of-scope block:
   - Query: `"What dose of warfarin should I take daily?"`
   - Expected: red BLOCK badge, safe refusal message, no medical content

4. **BLOCK demo (evidence)** — show an evidence-insufficient block:
   - Query: `"Tell me about a very obscure compound"`
   - Expected: BLOCK with LOW_GROUNDING or LOW_COVERAGE rule tag

5. **Self-correction demo** — show the retry path:
   - Query: `"What is the mechanism of action of amlodipine?"`
   - Mode: `self_correcting`
   - Check "Details & debug" → correction_applied, guard_pass = 2

6. **Mode comparison** — switch between modes for the same query:
   - Show baseline (no retrieval context) vs rag vs self_correcting
   - Show grounding/coverage scores in the Details section

7. **Evaluation metrics** — scroll down on the app to show the batch evaluation table

---

## Artifact Directory Tree

```
artifacts/
└── demo2/
    ├── evidence_index.pkl              # BM25 evidence index (from build_index)
    ├── inference_baseline.jsonl        # Baseline batch inference outputs
    ├── inference_rag.jsonl             # RAG batch inference outputs
    ├── inference_self_correcting.jsonl # Self-correcting batch inference outputs
    ├── inference_outputs.jsonl         # Single-mode default output
    ├── metrics.json                    # Single-mode evaluation metrics
    ├── metrics_by_mode.json            # Cross-mode comparison metrics
    └── plots/
        ├── roc_curve.png               # ROC curve (grounding threshold sweep)
        ├── confusion_matrix.png        # ALLOW/BLOCK confusion matrix
        ├── grounding_distribution.png  # Grounding score histogram
        ├── latency_distribution.png    # Latency histogram
        └── latency_by_mode.png         # Mode latency comparison bar chart
```

---

## Troubleshooting

**"Evidence index not found"**
```bash
python -m project_demo.rag.build_index --from-parquet silver/silver_drugs_from_delta.parquet
```

**"Ollama not reachable" / connection refused**
```bash
ollama serve
```
Then re-open the app or reload the page. The sidebar status indicator will turn green when Ollama is reachable.

**"openai package not installed"**
```bash
pip install openai
```
Note: this is the OpenAI Python SDK used to talk to the local Ollama server — not the OpenAI cloud service. No API key or internet access is needed.

**Model not found / "pull model first"**
```bash
ollama pull mistral
```
Or in the app sidebar: type any model tag into the custom model field after pulling it locally.

**Streamlit "module not found" error**
Make sure you run from the repo root:
```bash
cd /path/to/MSDA-Guardrails-Project-main
streamlit run project_demo/rag/app.py
```

**All decisions are BLOCK**
- Check that the evidence index has content: `ls -lh artifacts/demo2/evidence_index.pkl`
- Lower the grounding threshold in the sidebar (τ = 0.3 instead of 0.45)
- Try a drug name that is in the silver dataset (e.g., Warfarin, Aspirin, Metformin)

**Inference is slow**
- `mistral` (7B) is the fastest model for CPU inference
- Try `llama3.2` (3B) for even faster but lower-quality outputs
- If you have a GPU, Ollama will use it automatically
