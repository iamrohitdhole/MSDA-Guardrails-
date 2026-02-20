from pathlib import Path
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
DELTA_OUT = ROOT / "lake" / "silver" / "silver_good_delta"

spark = (SparkSession.builder
    .appName("QueryDelta")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.jars.packages","io.delta:delta-spark_2.13:4.0.0")
    .getOrCreate())

df = spark.read.format("delta").load(str(DELTA_OUT))
print("rows:", df.count())
df.groupBy("indication").count().orderBy("count", ascending=False).show(10, truncate=False)

print("history:")
spark.sql(f"CREATE TABLE IF NOT EXISTS silver_good USING DELTA LOCATION '{DELTA_OUT.as_posix()}'")
spark.sql("DESCRIBE HISTORY silver_good").show(truncate=False)

spark.stop()
