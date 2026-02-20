import os
from pyspark.sql import SparkSession, functions as F, types as T

spark=(SparkSession.builder
 .appName("ingest_chat_queries_raw_parquet")
 .config("spark.hadoop.fs.s3a.endpoint","http://localhost:9000")
 .config("spark.hadoop.fs.s3a.path.style.access","true")
 .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID","minio"))
 .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY","minioStrongPass123"))
 .getOrCreate())

bootstrap = "localhost:9094"
topic = "chat-queries"

raw = (spark.readStream
  .format("kafka")
  .option("kafka.bootstrap.servers", bootstrap)
  .option("subscribe", topic)
  .option("startingOffsets","earliest")
  .load())

schema = T.StructType([
    T.StructField("question", T.StringType()),
    T.StructField("user_id",  T.StringType()),
    T.StructField("ts",       T.StringType())
])

json = raw.select(F.col("value").cast("string").alias("value"))
parsed = json.select(F.from_json("value", schema).alias("j")).select("j.*")

out_path   = "s3a://bronze/chat_queries_raw_v2"
checkpoint = "s3a://bronze/checkpoints/chat_queries_raw_parquet_v2"

query = (parsed
  .withColumn("dt", F.date_format(F.current_timestamp(),"yyyy-MM-dd"))
  .writeStream
  .format("parquet")
  .outputMode("append")
  .option("checkpointLocation", checkpoint)
  .option("path", out_path)
  .partitionBy("dt")        # added to partition by landing date
  .partitionBy("dt")        # added to partition by landing date
  .start())

query.awaitTermination()
