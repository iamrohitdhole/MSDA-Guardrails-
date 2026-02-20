from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer

ROOT = Path(__file__).resolve().parents[1]
DELTA_IN  = ROOT / "lake" / "silver" / "silver_good_delta"
DELTA_OUT = ROOT / "lake" / "feature" / "drugs_features_delta"

spark = (SparkSession.builder
    .appName("FeatureEngineeringDrugs")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
    .getOrCreate())

df = spark.read.format("delta").load(str(DELTA_IN))

# Basic numeric features
df = (df
    .withColumn("name_length", F.length("drug_name"))
    .withColumn("desc_length", F.size(F.split(F.col("description"), "\\s+")))
    .withColumn("has_toxicity_info", F.when(F.col("toxicity").isNotNull(), 1).otherwise(0))
)

# Keyword flags (simple substring match)
keywords = ["agonist", "inhibitor", "antagonist"]
for kw in keywords:
    df = df.withColumn(f"mech_{kw}", F.when(F.lower(F.col("mechanism")).contains(kw), 1).otherwise(0))

# Encode indication to numeric
indexer = StringIndexer(inputCol="indication", outputCol="indication_index", handleInvalid="keep")
df = indexer.fit(df).transform(df)

# Write engineered features as new Delta table
(df.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .save(str(DELTA_OUT))
)

print("Feature-engineered Delta written to:", DELTA_OUT)
print("\n=== SAMPLE: Feature-engineered rows ===")
df.select(
    "drug_id","drug_name","name_length","desc_length",
    "has_toxicity_info","indication_index",
    "mech_agonist","mech_inhibitor","mech_antagonist"
).show(5, truncate=False)

print("\n=== FEATURE SUMMARY ===")
summary = (
    df.select(
        F.mean("name_length").alias("avg_name_len"),
        F.mean("desc_length").alias("avg_desc_len"),
        F.sum("has_toxicity_info").alias("toxicity_records"),
        F.count("*").alias("total")
    )
).toPandas()
print(summary)

spark.stop()
