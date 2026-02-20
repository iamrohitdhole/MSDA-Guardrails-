# Demo 1: End-to-End System Architecture & Data Ingestion

Use this content for your **system architecture overview** and **data ingestion** slides.

---

## 1. End-to-End System Architecture Overview

The platform is a **medallion-style data pipeline** with ML and guardrails:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES & INGESTION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  DrugBank XML (file)     Kafka (raw_input)     Optional: Postgres / Chat          │
│         │                        │                        │                      │
│         ▼                        ▼                        ▼                      │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐              │
│  │  land_xml_   │         │ ingestion_   │         │  Other       │              │
│  │  raw.py /    │         │ dag          │         │  ingest      │              │
│  │  Airflow DAG │         │ (Kafka→S3)   │         │  scripts     │              │
│  └──────┬───────┘         └──────┬───────┘         └──────┬───────┘              │
└─────────┼────────────────────────┼────────────────────────┼───────────────────────┘
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  BRONZE LAYER (MinIO/S3)                                                          │
│  Buckets: raw, bronze                                                             │
│  Formats: Parquet (raw_xml_drugs_parquet), JSONL (silver/drugbank/, raw/)        │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SILVER LAYER (cleaned, deduplicated, typed)                                       │
│  - bronze_to_silver_drugs_polished.py → s3a://bronze/silver/drugs/ (Delta)        │
│  - quality_enforce_silver.py → good/bad split, summary CSV                         │
│  - promote_and_report.py → promoted Silver, optional s3a://bronze/silver/drugs_   │
│    clean/                                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ├──────────────────────────────────┬─────────────────────────────────────┐
          ▼                                  ▼                                     ▼
┌─────────────────────┐    ┌─────────────────────────────┐    ┌─────────────────────┐
│  GOLD (aggregations)│    │  ML / TRAINING PIPELINE      │    │  LLM ENRICHMENT     │
│  silver_to_gold_     │    │  prepare_splits.py           │    │  llm_enrich_        │
│  drugs.py           │    │  → train/val/test            │    │  drugbank.py        │
│  (mass bins, top     │    │  train_text_classifier.py   │    │  (Groq LLaMA)       │
│  indications)       │    │  train_text_classifier_rf.py │    │  → gold/drugbank_   │
│                     │    │  compare_models.py           │    │  llm/*.jsonl       │
└─────────────────────┘    └─────────────────────────────┘    └─────────────────────┘
```

**Components:**

| Component | Role |
|-----------|------|
| **MinIO (S3)** | Object store for raw, bronze, silver, gold (Parquet, JSONL, Delta). |
| **Kafka** | Optional streaming ingestion; `ingestion_dag` consumes and lands to bronze. |
| **Airflow** | Orchestration: DrugBank XML→Silver DAG, ingestion DAG, LLM debug DAG. |
| **Spark (PySpark)** | Batch ETL: XML→Bronze, Bronze→Silver (Delta), DQ, Silver→Gold. |
| **Python (sklearn, etc.)** | Train/test splits, TF-IDF + Logistic Regression, TF-IDF + Random Forest, comparison. |
| **LLM (Groq)** | Enrichment of DrugBank records; guardrails validation (required fields, hallucination checks). |

---

## 2. Data Ingestion: How Data Enters and Flows

### 2.1 Entry points

- **DrugBank XML**  
  - **Source:** Local file `project_demo/raw/full_database.xml` or MinIO `raw/ddi_xml/database.xml`.  
  - **Ingestion:**  
    - **Batch:** `land_xml_raw.py` reads XML with Spark, writes Parquet to `s3a://bronze/raw_xml_drugs_parquet/`.  
    - **Airflow:** DAG `drugbank_xml_to_silver` stream-parses XML from MinIO, writes a JSONL sample to `s3://bronze/silver/drugbank/`.

- **Kafka**  
  - **Source:** Topic `raw_input` (JSON messages).  
  - **Ingestion:** Airflow `ingestion_dag` consumes with `KafkaConsumer`, writes JSONL batches to `s3://bronze/raw/`, then a transform task writes cleaned Silver to `s3://bronze/silver/`.

### 2.2 Flow through the pipeline

1. **Raw → Bronze**  
   XML or Kafka events land in MinIO (raw/bronze buckets) as Parquet or JSONL.

2. **Bronze → Silver**  
   - **Drugs:** Spark job reads Bronze Parquet, cleans (primary ID, trim, cast), deduplicates by `drug_id`, writes Delta to `s3a://bronze/silver/drugs/` and optionally local Parquet.  
   - **Data quality:** `quality_enforce_silver.py` applies rule-based checks (missing name/ID, duplicate IDs, ID format, mass range, short text). Outputs: good/bad Parquet, summary CSV.

3. **Silver → Gold**  
   - **Analytics:** `silver_to_gold_drugs.py` builds aggregate tables (e.g. mass bins, top indications) as Delta in `s3a://bronze/gold/`.  
   - **ML:** `prepare_splits.py` reads Silver (e.g. `dq/out/silver_good.parquet`), produces stratified train/val/test under `dq/out/gold/samples/`.  
   - **LLM:** `llm_enrich_drugbank.py` reads latest DrugBank silver sample from MinIO, calls Groq, validates with guardrails, writes enriched JSONL to `s3://bronze/gold/drugbank_llm/`.

4. **Use across the pipeline**  
   - **Analytics/EDA:** Silver and Gold Delta/Parquet are used for stats and plots (e.g. `eda_demo_drugs.py`, Streamlit apps).  
   - **ML:** Gold samples feed the baseline (TF-IDF + Logistic Regression) and the new model (TF-IDF + Random Forest); `compare_models.py` produces side-by-side metrics.  
   - **Guardrails:** LLM outputs are validated (required fields, hallucination patterns); DQ scores and issues are stored with each enriched record.

---

## 3. One-Page Diagram (for slides)

You can paste this into a slide as a high-level flow:

**Data flow (simplified):**  
`XML / Kafka` → **Bronze (MinIO)** → **Silver (Spark + DQ)** → **Gold (aggregations, train/val/test, LLM enrichment)**.  
**ML:** Silver good → `prepare_splits` → train/val/test → **Baseline (TF-IDF + LR)** and **Random Forest (TF-IDF + RF)** → metrics and comparison.  
**Orchestration:** Airflow DAGs for XML→Silver, Kafka→Bronze→Silver, and LLM-related steps.

This document should be used together with **DEMO1_Model_Evaluation.md** and **DEMO1_Data_Analytics.md** for a complete Demo 1 presentation.
