from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import boto3
import json
import io
import uuid
import os


# ============================================================
# CONFIGURATION
# ============================================================

# MinIO endpoint INSIDE DOCKER NETWORK
# Airflow and MinIO are on the same Docker network, so we use the service name.
S3_ENDPOINT = "http://minio:9000"

# MinIO credentials
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minioStrongPass123"

# Bucket + paths
BUCKET = "bronze"
BRONZE_PREFIX = "raw/"
SILVER_PREFIX = "silver/"

# Kafka configuration
KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "raw_input"


# ============================================================
# HELPERS
# ============================================================

def get_minio_client():
    """Return a boto3 S3 client configured for MinIO."""
    print(f"[DEBUG] Creating MinIO client with endpoint: {S3_ENDPOINT}")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


# ============================================================
# TASK 1 — INGEST FROM KAFKA → BRONZE (RAW)
# ============================================================

def ingest_from_kafka_to_minio(**context):
    """Consume messages from Kafka and upload a JSONL batch into MinIO bronze/raw."""
    print(f"[INFO] Using MinIO endpoint: {S3_ENDPOINT}")

    # ---- Connect to Kafka ----
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=5000,  # exit if no messages
        )
    except NoBrokersAvailable:
        print("[ERROR] Kafka broker is not reachable from Airflow. Skipping ingestion.")
        return None

    messages = []
    for msg in consumer:
        messages.append(msg.value)
        if len(messages) >= 20:  # Limit for demo
            break

    if not messages:
        print("[INFO] No messages consumed from Kafka.")
        return None

    # ---- Upload to MinIO ----
    s3 = get_minio_client()

    body_bytes = ("\n".join(json.dumps(rec) for rec in messages)).encode("utf-8")

    object_key = (
        f"{BRONZE_PREFIX}batch_"
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex}.jsonl"
    )

    s3.upload_fileobj(io.BytesIO(body_bytes), BUCKET, object_key)

    print(f"[INFO] Wrote {len(messages)} records → s3://{BUCKET}/{object_key}")

    return object_key  # XCom for next task


# ============================================================
# TASK 2 — TRANSFORM RAW → SILVER
# ============================================================

def transform_raw_to_silver(**context):
    """Read Bronze JSONL, clean records, and write Silver JSONL."""
    ti = context["ti"]
    bronze_key = ti.xcom_pull(task_ids="kafka_to_minio_raw")

    if not bronze_key:
        print("[WARN] No bronze file returned; skipping transformation.")
        return None

    s3 = get_minio_client()

    # ---- Read Bronze JSONL ----
    obj = s3.get_object(Bucket=BUCKET, Key=bronze_key)
    content = obj["Body"].read().decode("utf-8")

    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "text" not in rec:
            continue

        clean_rec = {
            "id": rec.get("id"),
            "source": rec.get("source", "unknown"),
            "text": rec["text"],
            "text_length": len(rec["text"]),
            "processed_at": datetime.utcnow().isoformat(),
        }
        records.append(clean_rec)

    if not records:
        print(f"[WARN] No valid records found in {bronze_key}")
        return None

    # ---- Write Silver JSONL ----
    silver_body = "\n".join(json.dumps(r) for r in records).encode("utf-8")
    silver_key = (
        f"{SILVER_PREFIX}clean_"
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex}.jsonl"
    )

    s3.upload_fileobj(io.BytesIO(silver_body), BUCKET, silver_key)

    print(
        f"[INFO] Transformed {len(records)} records "
        f"→ s3://{BUCKET}/{silver_key}"
    )

    return silver_key


# ============================================================
# DAG DEFINITION
# ============================================================

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 12, 1),
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id="ingestion_dag",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False,
    tags=["bronze", "kafka", "minio", "silver"],
) as dag:

    ingest_task = PythonOperator(
        task_id="kafka_to_minio_raw",
        python_callable=ingest_from_kafka_to_minio,
        provide_context=True,
    )

    transform_task = PythonOperator(
        task_id="raw_to_silver_clean",
        python_callable=transform_raw_to_silver,
        provide_context=True,
    )

    ingest_task >> transform_task
