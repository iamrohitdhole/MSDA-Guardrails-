# Runbook: Full Pipeline + Three Airflow DAGs

This document records the **exact workflow** used to run the project from scratch: environment setup, Spark/Python pipeline, then Airflow DAGs.

---

## Part 1 — Environment setup

### Conda and Python

```bash
conda create -n datapipeline python=3.10 -y
conda activate datapipeline
```

### Pip dependencies

```bash
pip install pyspark==3.5.3
pip install delta-spark==3.2.1
pip install minio
pip install apache-airflow
pip install groq
pip install pydantic
pip install kafka-python
pip install streamlit
pip install pandas matplotlib
```

### Java (for Spark)

Add to `~/.zshrc` (e.g. with `nano ~/.zshrc`):

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
export PATH=$JAVA_HOME/bin:$PATH
```

Then:

```bash
source ~/.zshrc
echo $JAVA_HOME   # should print Java 11 path
```

---

## Part 2 — Infrastructure

From the **project root**, start MinIO (and Kafka if you use the ingestion DAG):

```bash
cd /Users/artemis/Downloads/MSDA-GUardrails-Project-main
docker compose up -d
```

- Create buckets **`raw`** and **`bronze`** in the MinIO console (http://localhost:9001).
- For the DrugBank Airflow DAG, upload XML to **`raw/ddi_xml/database.xml`** (e.g. from `project_demo/raw/full_database.xml`).

---

## Part 3 — Spark/Python pipeline (run from project root)

Use the **project root** as working directory and set `PYTHONPATH` so that `project_demo` packages resolve:

```bash
cd /Users/artemis/Downloads/MSDA-GUardrails-Project-main
export PYTHONPATH=/Users/artemis/Downloads/MSDA-GUardrails-Project-main
```

### 3.1 Land XML → Bronze (Parquet in MinIO)

```bash
python -m project_demo.raw.land_xml_raw
# Or with Spark explicitly:
# spark-submit --packages io.delta:delta-spark_2.12:3.2.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,com.databricks:spark-xml_2.12:0.17.0 project_demo/raw/land_xml_raw.py
```

### 3.2 Bronze → Silver (Delta + local Parquet)

```bash
python -m project_demo.silver.bronze_to_silver_drugs_polished
```

(Optional flatten step: `python -m project_demo.silver.bronze_to_silver_drugs_flatten`.)

### 3.3 Read Silver (verify)

```bash
python -m project_demo.silver.read_silver_drugs
```

### 3.4 Data quality: Silver → good/bad split + summary

**Input:** Silver parquet (local or S3). Output: `artifacts/dq/` (good/bad parquet, summary CSV).

```bash
spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
  project_demo/dq/quality_enforce_silver.py \
  --input-path /Users/artemis/Downloads/MSDA-GUardrails-Project-main/project_demo/dq/out/silver_good.parquet
```

If Silver is in MinIO, use e.g. `--input-path s3a://bronze/silver/drugs/` (Delta).

### 3.5 Promote good Silver + DQ report (optional: publish to S3)

**Variant A — Local only (no S3 publish):**

From **project_demo/dq** (so relative paths like `artifacts/dq/` resolve as in the script), or from project root with absolute paths:

```bash
python promote_and_report.py \
  --good-path /Users/artemis/Downloads/MSDA-GUardrails-Project-main/artifacts/dq/good_silver.parquet \
  --bad-path /Users/artemis/Downloads/MSDA-GUardrails-Project-main/artifacts/dq/bad_silver.parquet \
  --summary-csv /Users/artemis/Downloads/MSDA-GUardrails-Project-main/artifacts/dq/summary.csv
```

**Variant B — With Spark + publish to MinIO (Delta):**

```bash
spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
  project_demo/dq/promote_and_report.py \
  --good-path /Users/artemis/Downloads/MSDA-GUardrails-Project-main/artifacts/dq/good_silver.parquet \
  --bad-path /Users/artemis/Downloads/MSDA-GUardrails-Project-main/artifacts/dq/bad_silver.parquet \
  --silver-local /Users/artemis/Downloads/MSDA-GUardrails-Project-main/silver/silver_drugs.parquet \
  --publish-s3 \
  --s3-delta-path s3a://bronze/silver/drugs_clean/
```

### 3.6 LLM enrichment (Silver sample → Gold JSONL in MinIO)

Uses the latest DrugBank silver sample in `s3://bronze/silver/drugbank/` (e.g. produced by the Airflow DAG or uploaded manually). Optional: `GROQ_API_KEY` for real LLM calls.

```bash
export S3_ENDPOINT=http://localhost:9000
# export GROQ_API_KEY=your_key   # optional
python -m project_demo.scripts.llm_enrich_drugbank
```

---

## Part 4 — Airflow DAGs

After the pipeline above (and with MinIO running and buckets/XML in place):

```bash
cd /Users/artemis/Downloads/MSDA-GUardrails-Project-main/airflow
docker compose up -d
```

- **Airflow UI:** http://localhost:8081

Then trigger these three DAGs manually:

1. **test_dag** — Sanity check (EmptyOperator).
2. **drugbank_xml_to_silver** — Reads `raw/ddi_xml/database.xml`, writes silver JSONL sample to `bronze/silver/drugbank/`.
3. **llm_enrich_drugbank_dag** — Debug paths (BashOperator). Full LLM enrichment is the `python -m project_demo.scripts.llm_enrich_drugbank` step above.

If you also use **ingestion_dag** (Kafka → Bronze → Silver), ensure Airflow can reach `minio:9000` and `kafka:9092` (e.g. same Docker network).

---

## Summary order

| Step | Command / action |
|------|-------------------|
| 1 | Conda env + pip installs + Java in `~/.zshrc` |
| 2 | `docker compose up -d` (project root) |
| 3 | Create buckets `raw`, `bronze`; upload XML to `raw/ddi_xml/database.xml` |
| 4 | `python -m project_demo.raw.land_xml_raw` |
| 5 | `python -m project_demo.silver.bronze_to_silver_drugs_polished` |
| 6 | (Optional) `python -m project_demo.silver.read_silver_drugs` |
| 7 | `spark-submit ... project_demo/dq/quality_enforce_silver.py --input-path <silver_parquet_or_s3>` |
| 8 | `python promote_and_report.py ...` or `spark-submit ... promote_and_report.py ... --publish-s3 ...` |
| 9 | (Optional) `python -m project_demo.scripts.llm_enrich_drugbank` |
| 10 | `cd airflow && docker compose up -d`; trigger **test_dag**, **drugbank_xml_to_silver**, **llm_enrich_drugbank_dag** |

---

## Notes

- **Paths:** Adjust `/Users/artemis/Downloads/MSDA-GUardrails-Project-main` if your project root is elsewhere.
- **quality_enforce_silver.py:** Default input in the script is `project_demo/dq/out/silver_good.parquet`; use `--input-path` to point to your Silver output (local parquet or `s3a://bronze/silver/drugs/`).
- **promote_and_report.py:** Can be run with plain `python` (no Spark) for the local-only variant; use `spark-submit` when using `--publish-s3`.
