import os
from pyspark.sql import SparkSession, functions as F

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB",   "appdb")
PG_USER = os.getenv("PG_USER", "appuser")
PG_PASS = os.getenv("PG_PASS", "apppass")
PG_TABLE= os.getenv("PG_TABLE","public.orders")  # change to your table
WATERMARK_PATH = "s3a://bronze/checkpoints/pg_orders_raw_watermark/state.parquet"

spark = (SparkSession.builder
    .appName("ingest_pg_orders_raw_parquet")
    .config("spark.hadoop.fs.s3a.endpoint","http://localhost:9000")
    .config("spark.hadoop.fs.s3a.path.style.access","true")
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID","minio"))
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY","minioStrongPass123"))
    .getOrCreate())

jdbc_url = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
jdbc_opts = {
    "url": jdbc_url,
    "user": PG_USER,
    "password": PG_PASS,
    "driver": "org.postgresql.Driver",
    "fetchsize": "10000",
}

# get last watermark if exists
def load_watermark():
    try:
        wm = spark.read.parquet(WATERMARK_PATH).agg(F.max("max_updated_at")).first()[0]
        return wm
    except Exception:
        return None

last_wm = load_watermark()
if last_wm:
    predicate = f"(SELECT * FROM {PG_TABLE} WHERE updated_at > TIMESTAMP '{last_wm}') t"
else:
    predicate = f"(SELECT * FROM {PG_TABLE}) t"   # bootstrap full if no watermark

jdbc_opts["dbtable"] = predicate

df = spark.read.format("jdbc").options(**jdbc_opts).load()

# add landing partitions
df = df.withColumn("dt", F.date_format(F.current_timestamp(), "yyyy-MM-dd")) \
       .withColumn("updated_date", F.to_date("updated_at"))

out_path = "s3a://bronze/orders_raw/"
(df.write
   .mode("append")
   .partitionBy("dt")
   .parquet(out_path))

# persist new watermark
mx = df.agg(F.max("updated_at").alias("max_updated_at"))
(mx.write.mode("overwrite").parquet(WATERMARK_PATH))

print("Postgres → Bronze done. Rows:", df.count())
