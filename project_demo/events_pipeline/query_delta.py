from pyspark.sql import SparkSession

ENDPOINT="http://localhost:9000"
ACCESS_KEY="minio"
SECRET_KEY="minioStrongPass123"
SILVER_PATH="s3a://bronze/silver/events_delta"

spark = (SparkSession.builder.appName("query-delta")
  .config("spark.jars.packages","io.delta:delta-spark_2.12:3.2.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
  .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
  .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
  .config("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
  .config("spark.hadoop.fs.s3a.endpoint",ENDPOINT)
  .config("spark.hadoop.fs.s3a.path.style.access","true")
  .config("spark.hadoop.fs.s3a.connection.ssl.enabled","false")
  .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
  .config("spark.hadoop.fs.s3a.access.key",ACCESS_KEY)
  .config("spark.hadoop.fs.s3a.secret.key",SECRET_KEY)
  .getOrCreate())

df = spark.read.format("delta").load(SILVER_PATH)
df.createOrReplaceTempView("events")

print("Row count:", spark.sql("select count(*) c from events").collect()[0]["c"])
spark.sql("select topic, value, ts, dt from events order by ts desc limit 5").show(truncate=False)
spark.stop()
