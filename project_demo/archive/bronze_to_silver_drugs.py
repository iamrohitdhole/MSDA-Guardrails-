from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("bronze_to_silver_drugs")
         .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
         .config("spark.hadoop.fs.s3a.access.key", "minio")
         .config("spark.hadoop.fs.s3a.secret.key", "minioStrongPass123")
         .config("spark.hadoop.fs.s3a.path.style.access", "true")
         .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
         .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
         .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                 "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
         .getOrCreate())

bronze_path = "s3a://bronze/raw_xml_drugs_parquet/"

df = spark.read.parquet(bronze_path)

print("Bronze row count:", df.count())
df.printSchema()
df.show(5, truncate=False)
