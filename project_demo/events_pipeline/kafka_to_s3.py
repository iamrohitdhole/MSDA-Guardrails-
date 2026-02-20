import json, os
from datetime import datetime, timezone
from confluent_kafka import Consumer
import boto3

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9094")
TOPIC = os.getenv("TOPIC", "demo-events")

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minioStrongPass123",
    region_name="us-east-1",
)
BUCKET = "bronze"
PREFIX = "kafka_ingest/"

c = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id": "cli-s3-1",
    "auto.offset.reset": "earliest",
})

c.subscribe([TOPIC])
print(f"Consuming 1 message from {TOPIC} on {KAFKA_BOOTSTRAP} ...")
msg = c.poll(timeout=10.0)

if msg is None:
    print("No message received (timeout).")
elif msg.error():
    print(f"Kafka error: {msg.error()}")
else:
    val = msg.value().decode("utf-8", errors="replace")
    record = {
        "topic": msg.topic(),
        "partition": msg.partition(),
        "offset": msg.offset(),
        "value": val,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    body = (json.dumps(record) + "\n").encode("utf-8")
    key = f"{PREFIX}{msg.topic()}-{msg.partition()}-{msg.offset()}.jsonl"
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)
    print(f"Wrote s3://{BUCKET}/{key}")

c.close()
