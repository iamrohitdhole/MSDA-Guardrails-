from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("read_silver_drugs")

    # 🔥 ADD THIS BLOCK (Delta + S3A dependencies)
    .config(
        "spark.jars.packages",
        ",".join([
            "io.delta:delta-spark_2.12:3.2.1",
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        ])
    )

    # MinIO / S3A configs
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minio")
    .config("spark.hadoop.fs.s3a.secret.key", "minioStrongPass123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

    # Delta configs
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    .getOrCreate()
)

silver_path = "s3a://silver/drugs_flatten_delta/"  # 🔥 FIXED PATH

df_silver = spark.read.format("delta").load(silver_path)

print("Silver row count =", df_silver.count())
df_silver.printSchema()
df_silver.show(10, truncate=False)

spark.stop()