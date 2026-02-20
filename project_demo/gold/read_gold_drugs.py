from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("read_gold_drugs")
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

mass_path = "s3a://bronze/gold/drugs_summary/"
inds_path = "s3a://bronze/gold/top_indications/"

df_mass = spark.read.format("delta").load(mass_path)
df_inds = spark.read.format("delta").load(inds_path)

print("\n=== GOLD: Mass bins ===")
df_mass.orderBy("mass_bin").show(truncate=False)

print("\n=== GOLD: Top 10 indications ===")
df_inds.orderBy(df_inds.drug_count.desc()).show(truncate=False)

print(f"\n✅ Read Gold: {mass_path} and {inds_path}")
spark.stop()
