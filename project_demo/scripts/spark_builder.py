from pyspark.sql import SparkSession
from project_demo.scripts.config_io import MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET


def build_spark(app_name: str = "MSDA-Guardrails") -> SparkSession:
    """
    Spark configured for:
      - Delta Lake
      - MinIO (S3A)
      - XML parsing
    Compatible with Spark 3.5.x (Scala 2.12)
    """

    packages = [
        # Delta Lake (Spark 3.5 -> Scala 2.12)
        "io.delta:delta-spark_2.12:3.2.1",

        # MinIO / S3A
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",

        # XML support
        "com.databricks:spark-xml_2.12:0.17.0"
    ]

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", ",".join(packages))

        # Delta support
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

        # MinIO / S3A configs
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark