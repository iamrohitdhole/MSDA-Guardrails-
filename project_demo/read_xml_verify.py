import os
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .appName("xml_read_verify")
    .config("spark.hadoop.fs.s3a.endpoint","http://localhost:9000")
    .config("spark.hadoop.fs.s3a.path.style.access","true")
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID","minio"))
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY","minioStrongPass123"))
    .getOrCreate())

xml_path = "s3a://raw/ddi_xml/database.xml"

# Try common DrugBank-like repeating tag first.
df = (spark.read
      .format("com.databricks.spark.xml")
      .option("rowTag","drug")           # <-- we'll change this if needed
      .option("samplingRatio","0.005")
      .load(xml_path))

print("XML read OK. Counting rows... (may take a moment)")
cnt = df.count()
print(f"Row count: {cnt}\n")

print("Schema (first levels):")
df.printSchema()
