from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode_outer


def build_spark(app_name: str = "bronze_to_silver_drugs_flatten") -> SparkSession:
    """
    Spark session builder with:
      - MinIO (S3A) support
      - Delta Lake support
    Compatible with Spark 3.5.x (Scala 2.12).
    """
    packages = [
        # Delta Lake (use Scala 2.12 artifact for Spark 3.5.x)
        "io.delta:delta-spark_2.12:3.2.1",
        # S3A support (MinIO)
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    ]

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", ",".join(packages))

        # ---- MinIO / S3A configs ----
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minio")
        .config("spark.hadoop.fs.s3a.secret.key", "minioStrongPass123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

        # ---- Delta Lake configs ----
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

        # Optional: fewer noisy logs
        .config("spark.ui.showConsoleProgress", "true")

        .getOrCreate()
    )

    return spark


def main():
    spark = build_spark()

    # Input (bronze) and output (silver) paths
    bronze_path = "s3a://bronze/raw_xml_drugs_parquet/"
    # ✅ Write to the SILVER bucket (not inside bronze)
    silver_path = "s3a://silver/drugs_flatten_delta/"

    print(f"📥 Reading Bronze parquet from: {bronze_path}")
    df = spark.read.parquet(bronze_path)

    # Flatten drugbank-id and select key fields
    df_flat = (
        df
        .withColumn("drugbank_id", explode_outer("drugbank-id"))
        .select(
            col("drugbank_id._VALUE").alias("drug_id"),
            col("name").alias("drug_name"),
            col("average-mass").alias("avg_mass"),
            col("description").alias("description"),
            col("indication").alias("indication"),
            col("mechanism-of-action").alias("mechanism"),
            col("toxicity").alias("toxicity"),
    
        )
        .dropDuplicates(["drug_id"])
        .filter(col("drug_id").isNotNull())
    )

    row_count = df_flat.count()
    print(f"✅ Silver (flattened) row count: {row_count}")
    df_flat.show(5, truncate=False)

    print(f"📤 Writing Silver Delta to: {silver_path}")
    (
        df_flat.write
        .format("delta")
        .mode("overwrite")
        .save(silver_path)
    )

    print(f"✅ Silver Delta table written successfully → {silver_path}")
    spark.stop()


if __name__ == "__main__":
    main()
