from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import io
import json
import xml.etree.ElementTree as ET
import uuid

# ------------------------
# MinIO / S3 Configuration
# ------------------------

MINIO_ENDPOINT = "http://host.docker.internal:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minioStrongPass123"

# XML lives here (we confirmed this via the Python REPL)
BUCKET = "raw"
XML_KEY = "ddi_xml/database.xml"

# Silver output will go here
SILVER_BUCKET = "bronze"
SILVER_PREFIX = "silver/drugbank/"

# Limit number of drugs parsed (demo / safety)
MAX_DRUGS = 200


def parse_drugbank_xml_to_silver(**context):
    """
    Stream-parse DrugBank XML from MinIO (raw bucket),
    extract a subset of fields, and write a JSONL sample
    to the bronze bucket under silver/drugbank/.
    """

    # 1. MinIO client (S3-compatible)
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

    # 2. Get XML object as a stream (do NOT read the whole thing into memory)
    obj = s3.get_object(Bucket=BUCKET, Key=XML_KEY)
    stream = obj["Body"]  # StreamingBody; file-like

    # 3. Stream-parse XML using iterparse
    # DrugBank uses a default namespace: xmlns="http://www.drugbank.ca"
    ns = "{http://www.drugbank.ca}"

    # iterparse with start/end so we can get the root and clear it as we go
    context_iter = ET.iterparse(stream, events=("start", "end"))
    _, root = next(context_iter)  # get root element

    records = []

    for event, elem in context_iter:
        # We only care when we finish a <drug> element
        if event == "end" and elem.tag == ns + "drug":
            # ---------- Extract fields from this <drug> ----------
            # All DrugBank IDs
            drugbank_ids = [
                el.text
                for el in elem.findall(ns + "drugbank-id")
                if el.text
            ]

            # Primary id (has attribute primary="true")
            primary_id_el = elem.find(f"{ns}drugbank-id[@primary='true']")
            if primary_id_el is not None and primary_id_el.text:
                primary_id = primary_id_el.text
            elif drugbank_ids:
                primary_id = drugbank_ids[0]
            else:
                primary_id = None

            # Name
            name_el = elem.find(ns + "name")
            name = name_el.text if name_el is not None else None

            # Description
            desc_el = elem.find(ns + "description")
            description = desc_el.text if desc_el is not None else None

            # Indication
            indication_el = elem.find(ns + "indication")
            indication = indication_el.text if indication_el is not None else None

            # Groups (approved, withdrawn, etc.)
            groups = [
                g.text
                for g in elem.findall(f"{ns}groups/{ns}group")
                if g.text
            ]

            records.append(
                {
                    "primary_id": primary_id,
                    "all_ids": drugbank_ids,
                    "name": name,
                    "description": description,
                    "indication": indication,
                    "groups": groups,
                }
            )

            # Clear the element to free memory
            elem.clear()
            root.clear()

            # Stop once we've collected enough drugs
            if len(records) >= MAX_DRUGS:
                break

    if not records:
        print("No drug records parsed from XML.")
        return

    # 4. Convert parsed records to JSONL
    body = "\n".join(json.dumps(r) for r in records)
    body_bytes = body.encode("utf-8")

    # 5. Build a unique object key for silver output
    object_key = (
        f"{SILVER_PREFIX}drugbank_sample_"
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex}.jsonl"
    )

    # 6. Upload JSONL file to MinIO (bronze bucket)
    s3.upload_fileobj(io.BytesIO(body_bytes), SILVER_BUCKET, object_key)

    print(
        f"Parsed {len(records)} drug records from "
        f"s3://{BUCKET}/{XML_KEY} -> s3://{SILVER_BUCKET}/{object_key}"
    )

    return object_key


# ------------------------
# DAG Definition
# ------------------------

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 12, 1),
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    "drugbank_xml_to_silver",
    default_args=default_args,
    schedule_interval=None,  # manual trigger for now
    catchup=False,
    tags=["drugbank", "xml", "silver"],
) as dag:

    xml_to_silver_task = PythonOperator(
        task_id="xml_to_silver_sample",
        python_callable=parse_drugbank_xml_to_silver,
        provide_context=True,
    )
