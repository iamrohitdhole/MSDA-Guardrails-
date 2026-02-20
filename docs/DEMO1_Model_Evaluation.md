# Demo 1: Model Evaluation — Setup, Metrics, and Side-by-Side Comparison

Use this for **model evaluation** and **earlier ML results** in your demo and to meet the rubric (training/test data, ML results, evaluation methods).

---

## 1. Experimental Setup

### 1.1 Task

- **Input:** Drug description text (`description` column).  
- **Output:** Predicted label (`mechanism` column).  
- **Type:** Multi-class text classification.

### 1.2 Data

- **Source:** Silver layer after data quality (good rows).  
- **Splits:** Produced by `prepare_splits.py` (stratified by `indication`, 70% train / 15% val / 15% test, rare classes pooled or sent to train).  
- **Location:** `project_demo/dq/out/gold/samples/`  
  - `train_sample.csv`, `val_sample.csv`, `test_sample.csv`  
- **Manifest:** `project_demo/dq/out/gold/split_manifest.json` (row counts, overlap checks, class distribution).  
- **No leakage:** `overlap_checks` in the manifest confirm zero ID overlap between train/val/test.

### 1.3 Features

- **Vectorization:** TF-IDF (same for both models).  
  - `max_features=20000`, `ngram_range=(1, 2)`.  
  - Fit on train (and val for feature space); test transformed only with the fitted vectorizer.

### 1.4 Models (two, for comparison)

| Model | Algorithm | Purpose |
|-------|------------|---------|
| **Baseline** | TF-IDF + **Logistic Regression** | Fast, interpretable baseline (existing). |
| **New model** | TF-IDF + **Random Forest** | Alternative model for comparison; same features, different decision boundary. |

### 1.5 Evaluation protocol

- **Metrics:** Accuracy, macro Precision, macro Recall, macro F1, ROC-AUC (macro, when computable).  
- **Holdout:** All metrics reported on the **test set** only.  
- **Artifacts:**  
  - Baseline: `model_eval_baseline.json`, `confusion_matrix_baseline.png`.  
  - RF: `model_eval_rf.json`, `confusion_matrix_rf.png`.  
  - Comparison: `model_comparison.json`, `model_comparison.txt` (side-by-side and “what changed”).

---

## 2. Evaluation Metrics (for each model)

- **Accuracy:** Proportion of correct predictions.  
- **Precision (macro):** Average precision per class (unweighted).  
- **Recall (macro):** Average recall per class (unweighted).  
- **F1-score (macro):** Average F1 per class (unweighted).  
- **ROC-AUC (macro):** One-vs-rest macro average when ≥2 classes and probabilities available; otherwise not computed.

These are the same metrics used in the rubric (accuracy, etc.) and are saved in the JSON files for both models.

---

## 3. How to Reproduce Results

From project root (with `PYTHONPATH` set):

```bash
# 1. Train baseline (TF-IDF + Logistic Regression)
python -m project_demo.scripts.train_text_classifier

# 2. Train new model (TF-IDF + Random Forest)
python -m project_demo.scripts.train_text_classifier_rf

# 3. Side-by-side comparison
python -m project_demo.scripts.compare_models
```

Outputs:

- `project_demo/dq/out/gold/model_eval_baseline.json`  
- `project_demo/dq/out/gold/model_eval_rf.json`  
- `project_demo/dq/out/gold/model_comparison.json`  
- `project_demo/dq/out/gold/model_comparison.txt`  
- `project_demo/dq/out/gold/confusion_matrix_baseline.png`  
- `project_demo/dq/out/gold/confusion_matrix_rf.png`

---

## 4. Side-by-Side Comparison (template for slides)

After running the three commands above, open `model_comparison.json` or `model_comparison.txt` and use the numbers for your slides. Template:

| Metric | Baseline (TF-IDF + LR) | New model (TF-IDF + RF) | Difference (RF − Baseline) |
|--------|------------------------|--------------------------|----------------------------|
| Accuracy | *from model_eval_baseline.json* | *from model_eval_rf.json* | *from model_comparison.json* |
| Precision (macro) | … | … | … |
| Recall (macro) | … | … | … |
| F1 (macro) | … | … | … |
| ROC-AUC (macro) | … | … | … |

**What changed (narrative for presentation):**

- **Baseline (Logistic Regression):** Linear model on TF-IDF features; fast and interpretable; good for establishing a performance floor.  
- **Random Forest:** Same TF-IDF input; ensemble of trees; can capture non-linear patterns; may improve or decrease metrics depending on data and class balance.  
- Use the “what_changed” section in `model_comparison.json` (or the summary in `model_comparison.txt`) to state which model is better on which metric and to explain trade-offs (e.g. accuracy vs interpretability, training time).

---

## 5. Rubric Alignment

- **Training and test data:** Demonstrated via `prepare_splits.py`, `split_manifest.json`, and the use of `train_sample.csv` / `test_sample.csv` in both models.  
- **Data pre-processing:** Shown in the pipeline (Silver DQ, good/bad split, then splits from good Silver).  
- **On-going ML results:** Shown by running both `train_text_classifier.py` and `train_text_classifier_rf.py` and displaying metrics and confusion matrices.  
- **Evaluation methods and metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix; all documented above and in the JSON/PNG outputs.

Use the saved JSON/PNG files and the comparison script output as evidence in your demo and video.
