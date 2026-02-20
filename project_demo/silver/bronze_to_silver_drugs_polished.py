from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, size, trim, current_timestamp
import os

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------

# Target MinIO S3 paths
BRONZE_S3_PATH = "s3a://bronze/raw_xml_drugs_parquet/"
SILVER_S3_PATH = "s3a://bronze/silver/drugs/"

# Optional local backup (Parquet)
LOCAL_SILVER_PATH = "silver/silver_drugs.parquet"

# ------------------------------------------------------------------------------
#  SPARK SESSION (MinIO + Delta)
# ------------------------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("bronze_to_silver_drugs_polished")
    .config(
    "spark.jars.packages",
    ",".join([
        "io.delta:delta-spark_2.12:3.2.1",
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262"
    ])
    )
    # ---- MinIO connection ----
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minio")
    .config("spark.hadoop.fs.s3a.secret.key", "minioStrongPass123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    )
    # ---- Delta Lake support ----
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------------------
#  READ BRONZE
# ------------------------------------------------------------------------------

print(f"[INFO] Reading Bronze data from {BRONZE_S3_PATH}")
df = spark.read.parquet(BRONZE_S3_PATH)

bronze_count = df.count()
print(f"[CLEANING] Bronze row count = {bronze_count}")

# ------------------------------------------------------------------------------
#  CLEANING & TRANSFORMATION
# ------------------------------------------------------------------------------

# Extract the primary DrugBank ID (if available), otherwise fallback to first element
df = df.withColumn("primary_ids", expr("filter(`drugbank-id`, x -> x._primary = true)"))
df = df.withColumn(
    "drug_id",
    when(size("primary_ids") > 0, col("primary_ids")[0]["_VALUE"])
    .otherwise(col("drugbank-id")[0]["_VALUE"])
)

# Select key fields and cast data types
df_flat = (
    df.select(
        col("drug_id"),
        trim(col("name")).alias("drug_name"),
        col("average-mass").cast("double").alias("avg_mass"),
        col("description"),
        col("indication"),
        col("mechanism-of-action").alias("mechanism"),
        col("toxicity"),

    )
    .withColumn("ingest_dt", current_timestamp())
)

before_dedup = df_flat.count()
print(f"[CLEANING] Before deduplication = {before_dedup}")

# Remove duplicate drug entries by ID
df_dedup = df_flat.dropDuplicates(["drug_id"])

after_dedup = df_dedup.count()
print(f"[CLEANING] After deduplication = {after_dedup}")
print(f"[CLEANING] {before_dedup - after_dedup} duplicate rows removed")

# Preview first few cleaned records
print("[PREVIEW] Showing sample of polished Silver data:")
df_dedup.show(10, truncate=False)

# ------------------------------------------------------------------------------
#  WRITE SILVER (to both MinIO + local)
# ------------------------------------------------------------------------------

print(f"[WRITE] Writing Silver data to S3 (Delta): {SILVER_S3_PATH}")
df_dedup.write.format("delta").mode("overwrite").save(SILVER_S3_PATH)
print("✅ Polished Silver Delta table written successfully to MinIO")

# Optionally also save local copy (for testing / quick access)
os.makedirs(os.path.dirname(LOCAL_SILVER_PATH), exist_ok=True)
df_dedup.write.mode("overwrite").parquet(LOCAL_SILVER_PATH)
print(f"✅ Local Parquet copy written to {LOCAL_SILVER_PATH}")

# ------------------------------------------------------------------------------
#  SUMMARY LOG
# ------------------------------------------------------------------------------

print("─────────────────────────────────────────────────────────────")
print("[SUMMARY]")
print(f" Bronze Input Count : {bronze_count}")
print(f" Silver Output Count: {after_dedup}")
print(f" Duplicates Removed : {before_dedup - after_dedup}")
print(f" Output (S3 Delta)  : {SILVER_S3_PATH}")
print(f" Output (Local)     : {LOCAL_SILVER_PATH}")
print("─────────────────────────────────────────────────────────────")

spark.stop()
print("🏁 Spark job complete — Silver dataset ready for downstream DQ/EDA.")
