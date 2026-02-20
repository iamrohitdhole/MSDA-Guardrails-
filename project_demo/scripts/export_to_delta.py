from pathlib import Path
from pyspark.sql import SparkSession

# --- paths ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # project_demo/
PARQUET_IN = ROOT / "dq" / "out" / "silver_good.parquet"
DELTA_OUT  = ROOT / "lake" / "silver" / "silver_good_delta"

# --- Spark WITH Delta extensions & catalog (local only) ---
spark = (
    SparkSession.builder
    .appName("ExportSilverToDelta")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    # IMPORTANT: bring Delta Lake jars
    .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
    .config("spark.sql.warehouse.dir", str(ROOT / "artifacts" / "spark-warehouse"))
    .getOrCreate()
)

# --- IO ---
df = spark.read.parquet(str(PARQUET_IN))
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(str(DELTA_OUT))
)

# --- sanity checks ---
delta_df = spark.read.format("delta").load(str(DELTA_OUT))
print("Delta row count:", delta_df.count())
delta_df.printSchema()

# Optional: register and show history
spark.sql(f"CREATE TABLE IF NOT EXISTS silver_good USING DELTA LOCATION '{DELTA_OUT.as_posix()}'")
spark.sql("DESCRIBE HISTORY silver_good").show(truncate=False)

spark.stop()
