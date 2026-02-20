import os
from pyspark.sql import SparkSession

spark=(SparkSession.builder
 .appName("read_parquet_verify")
 .config("spark.hadoop.fs.s3a.endpoint","http://localhost:9000")
 .config("spark.hadoop.fs.s3a.path.style.access","true")
 .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID","minio"))
 .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY","minioStrongPass123"))
 .getOrCreate())

df = spark.read.parquet("s3a://bronze/raw_xml_drugs_parquet")

print("Rows:", df.count())
print("Columns:", len(df.columns))
df.select("name","cas-number").show(10, truncate=False)
