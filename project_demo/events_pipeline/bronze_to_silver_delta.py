from pyspark.sql import SparkSession, functions as F

# S3/MinIO config
ENDPOINT = "http://localhost:9000"
ACCESS_KEY = "minio"
SECRET_KEY = "minioStrongPass123"

BRONZE_PATH = "s3a://bronze/kafka_ingest/*.jsonl"
SILVER_PATH = "s3a://bronze/silver/events_delta"

spark = (
    SparkSession.builder.appName("bronze-to-silver-delta")
    # Load Delta + Hadoop S3A jars in one go
    .config(
        "spark.jars.packages",
        ",".join([
            "io.delta:delta-spark_2.12:3.2.1",
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        ]),
    )
    # Enable Delta
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.S3SingleDriverLogStore")
    # S3A settings for MinIO
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.endpoint", ENDPOINT)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY)
    .getOrCreate()
)

# Read bronze JSONL from MinIO
df = spark.read.json(BRONZE_PATH)

# Simple transform: parse date, keep useful columns
df2 = (
    df
    .withColumn("dt", F.to_date(F.col("ts")))
    .withColumn("ingested_at", F.current_timestamp())
)

# Write as Delta, partitioned by dt
(df2.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("dt")
    .save(SILVER_PATH))

print("Wrote Delta table to:", SILVER_PATH)

# Quick sanity read
print("Rows:", spark.read.format("delta").load(SILVER_PATH).count())

spark.stop()
