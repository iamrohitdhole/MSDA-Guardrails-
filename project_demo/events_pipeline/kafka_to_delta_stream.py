from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

KAFKA_BOOTSTRAP = "localhost:9094"   # use 9092 if running inside the container
TOPIC = "demo-events"
DELTA_PATH = "s3a://bronze/stream/events_delta/"
CHKPT = "s3a://bronze/_chk/events_delta_ckpt"

schema = StructType([
    StructField("id", LongType(), True),
    StructField("ts", StringType(), True),
    StructField("source", StringType(), True),
    StructField("text", StringType(), True),
    StructField("amount", DoubleType(), True),
])

spark = (
    SparkSession.builder.appName("KafkaToDelta")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.endpoint","http://localhost:9000")
    .config("spark.hadoop.fs.s3a.access.key","minio")
    .config("spark.hadoop.fs.s3a.secret.key","minioStrongPass123")
    .config("spark.hadoop.fs.s3a.path.style.access","true")
    .config("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

raw = (spark.readStream.format("kafka")
       .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
       .option("subscribe", TOPIC)
       .option("startingOffsets", "latest")
       .load())

df = raw.select(from_json(col("value").cast("string"), schema).alias("j")).select("j.*")

query = (df.writeStream
         .format("delta")
         .option("checkpointLocation", CHKPT)
         .option("path", DELTA_PATH)
         .outputMode("append")
         .start())

spark.streams.awaitAnyTermination()
