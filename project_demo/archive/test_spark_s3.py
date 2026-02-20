from pyspark.sql import SparkSession

ENDPOINT = "http://localhost:9000"
ACCESS_KEY = "minio"
SECRET_KEY = "minioStrongPass123"
TARGET_PATH = "s3a://bronze/spark_test/parquet1"

spark = (
    SparkSession.builder.appName("spark-minio-test")
    .config(
        "spark.jars.packages",
        ",".join([
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        ]),
    )
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.endpoint", ENDPOINT)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY)
    .getOrCreate()
)

df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
df.write.mode("overwrite").parquet(TARGET_PATH)
df2 = spark.read.parquet(TARGET_PATH)
print("Rows:", df2.count())

spark.stop()
