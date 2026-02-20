# Demo 1: Checklist and Rubric Alignment

Use this to ensure your **demo**, **presentation slides**, and **video** meet all requirements.

---

## Deliverables

| Item | Where / How |
|------|-------------|
| **Demo presentation (PowerPoint) slides** | Build from the three DEMO1_*.md docs (Architecture, Model Evaluation, Data Analytics). |
| **Demo video** | Record a walkthrough of: slides, data prep, pipeline, analytics, ML results (baseline + new model), comparison, and current platform status. |

---

## What to demonstrate (rubric)

| Criterion | Pts | How to cover |
|-----------|-----|---------------|
| **Data pre-processing** | 2.5 | Show DQ step: good/bad split, summary CSV, Silver cleaning (dedup, schema). Use **DEMO1_Data_Analytics.md** + **DEMO1_Architecture_And_Data_Flow.md**. |
| **Training and test data** | 2.5 | Show `prepare_splits.py` output: train/val/test counts, `split_manifest.json`, no ID leakage. Use **DEMO1_Model_Evaluation.md**. |
| **Data analytics results** | 2.5 | Show EDA: mass distribution, top indications, daily counts; include **avg_mass_hist.png**, **top_indications.png**, **daily_counts.png**. Use **DEMO1_Data_Analytics.md**. |
| **On-going ML results** | 2.5 | Show **baseline** (TF-IDF + LR) and **new model** (TF-IDF + RF): metrics, confusion matrices, **side-by-side comparison** and “what changed.” Use **DEMO1_Model_Evaluation.md** + outputs of `compare_models.py`. |
| **Presentation** | 2.5 | Clear flow: architecture → data ingestion → pre-processing → train/test → analytics → ML results → comparison. Use visuals and short bullets. |

**Total: 12.5 pts**

---

## Presentation content (from instructions)

Your first demo presentation should include:

1. **End-to-end system architecture overview**  
   → **DEMO1_Architecture_And_Data_Flow.md** (diagram + components table).

2. **Data ingestion details**  
   → How data enters (XML, Kafka) and flows through Bronze → Silver → Gold; use the same doc.

3. **Model evaluation**  
   → Experimental setup, metrics, results; **side-by-side comparison** of baseline vs new model and **what changed**.  
   → **DEMO1_Model_Evaluation.md** + `model_comparison.json` / `model_comparison.txt`.

4. **Data analytics findings**  
   → Supported by graphs/visualizations (EDA plots, confusion matrices, DQ summary).  
   → **DEMO1_Data_Analytics.md**.

---

## Commands to run before recording

1. **Train baseline:**  
   `python -m project_demo.scripts.train_text_classifier`

2. **Train new model:**  
   `python -m project_demo.scripts.train_text_classifier_rf`

3. **Generate comparison:**  
   `python -m project_demo.scripts.compare_models`

4. **Optional (EDA plots):**  
   Run `project_demo/eda/eda_demo_drugs.py` (with Spark + MinIO) to refresh `avg_mass_hist.png`, `top_indications.png`, `daily_counts.png`.

Then use the generated JSON/PNG/TXT under `project_demo/dq/out/gold/` and EDA `plots/` in your slides and video.
