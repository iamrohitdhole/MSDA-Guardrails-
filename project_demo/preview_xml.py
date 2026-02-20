from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Minimal schema for a FAST preview (no full-file inference)
schema = StructType([
    # <drugbank-id primary="true">DB00001</drugbank-id>
    StructField("drugbank-id",
        StructType([StructField("_VALUE", StringType(), True)]), True),
    StructField("name", StringType(), True),
    StructField("cas-number", StringType(), True),
    StructField("state", StringType(), True),
    # <groups><group>approved</group>...</groups>
    StructField("groups",
        StructType([StructField("group", ArrayType(StringType()), True)]), True),
])

spark = (
    SparkSession.builder
    .appName("xml_preview_fast")
    .config(
        "spark.jars.packages",
        "com.databricks:spark-xml_2.12:0.17.0"
    )
    .getOrCreate()
)

# IMPORTANT: update path if your file still has a space in the name
xml_path = "raw/full_database.xml"  # use "raw/full database.xml" if you didn't rename

df = (
    spark.read.format("xml")
    .options(rowTag="drug", mode="PERMISSIVE", excludeAttribute=True)
    .schema(schema)                 # <-- critical: skip expensive schema inference
    .load(xml_path)
)

preview = df.select(
    F.col("`drugbank-id`._VALUE").alias("drugbank_id"),
    F.col("name"),
    F.col("`cas-number`").alias("cas_number"),
    F.col("state"),
    F.col("groups.group").alias("groups")
)

# LIMIT so Spark only touches a tiny slice
preview.limit(5).show(truncate=False)

# also drop a tiny CSV you can open as an 'AFTER' artifact
(preview
   .select("drugbank_id", "name", "cas_number", "state")
   .limit(20)
   .coalesce(1)
   .write.mode("overwrite")
   .option("header", "true")
   .csv("artifacts/after_preview_csv"))

spark.stop()
