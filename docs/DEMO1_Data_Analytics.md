# Demo 1: Data Analytics Findings and Visualizations

Use this for **data analytics results** and **graphs/visualizations** in your demo and slides.

---

## 1. Data analytics results (summary)

### 1.1 Silver / Gold layer

- **Silver:** Cleaned DrugBank records with `drug_id`, `drug_name`, `avg_mass`, `description`, `indication`, `mechanism`, `toxicity`, `ingest_dt`. Deduplicated by `drug_id`; written as Delta to MinIO and optionally as Parquet locally.  
- **Data quality:** Rule-based checks produce good/bad splits and a summary CSV (e.g. missing names, invalid IDs, mass out of range, short text). Use `artifacts/dq/summary.csv` and `dq/out/dq_report.md` for counts and samples.  
- **Gold:**  
  - Aggregate tables: drugs by mass bin, top indications (Delta in MinIO).  
  - Train/val/test splits: stratified by indication; see `dq/out/gold/split_manifest.json` for row counts and class distribution.  
  - LLM-enriched JSONL in MinIO with DQ scores and guardrail flags.

### 1.2 Key findings (for narrative)

- **Class imbalance:** Many rows have `__MISSING__` or `__RARE__` for indication/mechanism; stratified splits and rare-class handling in `prepare_splits.py` keep evaluation meaningful.  
- **Mass distribution:** `avg_mass` is right-skewed; EDA histograms and percentiles (p50/p90/p99) summarize it.  
- **Indications:** Long-tailed; top-10 indications bar chart and Gold “top indications” table show dominant use cases.  
- **ML:** Baseline (TF-IDF + LR) and Random Forest (TF-IDF + RF) both evaluated on the same test set; comparison table and “what changed” summarize performance differences.

---

## 2. Graphs and visualizations (where they live)

Include these in your slides and demo video.

### 2.1 EDA (Silver / Delta)

**Script:** `project_demo/eda/eda_demo_drugs.py`  
**Output directory:** `project_demo/eda/plots/` (or `./plots` when run from `eda/`)

| Figure | File | Description |
|--------|------|-------------|
| Distribution of drug average mass | `avg_mass_hist.png` | Histogram of `avg_mass` (sample). |
| Top 10 indications | `top_indications.png` | Horizontal bar chart of most frequent indications. |
| Daily ingest counts | `daily_counts.png` | Line plot of row count by `ingest_dt`. |

**How to regenerate:** From project root, with Silver Delta available (e.g. MinIO running and Silver written):

```bash
# If running from project_demo/eda with PYTHONPATH including project root
spark-submit --packages io.delta:delta-spark_2.12:3.2.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
  project_demo/eda/eda_demo_drugs.py
```

Plots will appear in the `plots` directory reported in the script output.

### 2.2 Data quality

- **Summary table:** `artifacts/dq/summary.csv` (or `project_demo/dq/out/summary.csv` depending on `quality_enforce_silver.py` args). Use for a small table or slide showing pass/fail counts per rule.  
- **DQ report:** `project_demo/dq/out/dq_report.md` — narrative and sample rows for good/bad; can be summarized in one slide.

### 2.3 Model evaluation (ML results)

**Location:** `project_demo/dq/out/gold/`

| Figure | File | Description |
|--------|------|-------------|
| Baseline confusion matrix | `confusion_matrix_baseline.png` | Heatmap: true vs predicted (description → mechanism). |
| Random Forest confusion matrix | `confusion_matrix_rf.png` | Same, for TF-IDF + RF model. |

**How to generate:** Run `train_text_classifier.py` and `train_text_classifier_rf.py` as in **DEMO1_Model_Evaluation.md**; the scripts write these PNGs.

### 2.4 Optional: correlation heatmap

**Script:** `project_demo/plots_useful/correlation_heatmap_code.py`  
If you have a numeric table (e.g. Silver with numeric columns), you can produce a correlation heatmap for one slide.

---

## 3. Suggested slide order (data analytics + visuals)

1. **Data pre-processing results**  
   - Screenshot or table from DQ summary (good/bad counts, key rules).  
   - Short mention of Silver schema and deduplication.

2. **Training and test data**  
   - Split counts and ratios from `split_manifest.json`.  
   - One sentence on stratified split and no ID leakage.

3. **Data analytics results**  
   - One slide: “Silver/Gold summary” (row counts, mass distribution, top indications).  
   - Next slide: include **avg_mass_hist.png** and **top_indications.png**.

4. **On-going ML results**  
   - Side-by-side metrics table (Baseline vs RF) from `model_comparison.json`.  
   - **confusion_matrix_baseline.png** and **confusion_matrix_rf.png** (side by side).  
   - One bullet: “What changed” from `model_comparison.txt`.

5. **Presentation / flow**  
   - Clear titles, short bullets, and a logical flow from raw → Bronze → Silver → Gold → ML and guardrails.

---

## 4. Rubric checklist (Data Analytics)

- **Data pre-processing:** Shown via DQ step, good/bad split, and Silver cleaning (RUNBOOK + this doc).  
- **Training and test data:** Shown via `prepare_splits.py`, manifest, and sample CSVs (DEMO1_Model_Evaluation + this doc).  
- **Data analytics results:** Shown via EDA stats, mass/indication findings, and the three EDA plots above.  
- **On-going ML results:** Shown via baseline + RF metrics and confusion matrices, plus comparison script output.  
- **Presentation:** Use architecture + data flow, model evaluation, and this analytics/visualization doc for clear, coherent slides and demo video.

Keep all figure paths relative to the project (e.g. `project_demo/dq/out/gold/confusion_matrix_baseline.png`) so slides and video stay consistent with the repo.
