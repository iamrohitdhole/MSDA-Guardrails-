from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count

# Spark session with MinIO + Delta
spark = (SparkSession.builder
         .appName("silver_to_gold_drugs")
         .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
         .config("spark.hadoop.fs.s3a.access.key", "minio")
         .config("spark.hadoop.fs.s3a.secret.key", "minioStrongPass123")
         .config("spark.hadoop.fs.s3a.path.style.access", "true")
         .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
         .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
         .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                 "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .getOrCreate())

silver_path = "s3a://bronze/silver/drugs/"
gold_path_summary = "s3a://bronze/gold/drugs_summary/"
gold_path_topindications = "s3a://bronze/gold/top_indications/"

# Read Silver
df = spark.read.format("delta").load(silver_path)
print("[GOLD] Silver row count =", df.count())

# 1. Drugs by avg_mass bins
df_mass_bins = (
    df.withColumn("mass_bin",
        when(col("avg_mass") < 100, "<100")
        .when((col("avg_mass") >= 100) & (col("avg_mass") <= 500), "100-500")
        .otherwise(">500")
    )
    .groupBy("mass_bin")
    .agg(count("*").alias("drug_count"))
)
df_mass_bins.show()

# 2. Top 10 indications
df_top_indications = (
    df.groupBy("indication")
      .agg(count("*").alias("drug_count"))
      .orderBy(col("drug_count").desc())
      .limit(10)
)
df_top_indications.show(truncate=False)

# Write Gold outputs
df_mass_bins.write.format("delta").mode("overwrite").save(gold_path_summary)
df_top_indications.write.format("delta").mode("overwrite").save(gold_path_topindications)

print(f"✅ Gold summary tables written to {gold_path_summary} and {gold_path_topindications}")
