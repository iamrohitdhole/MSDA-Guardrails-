from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, mean, stddev, expr, percentile_approx, when, length
)

# ---- Headless plotting ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Spark + MinIO + Delta
spark = (SparkSession.builder
         .appName("eda_demo_drugs")
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
df = spark.read.format("delta").load(silver_path).cache()
row_cnt = df.count()
print(f"\n[EDA] Loaded Silver table rows = {row_cnt}")

# ---------- Daily Stats ----------
print("\n=== Daily Stats (count / mean / std of avg_mass) ===")
daily_stats = (df.groupBy("ingest_dt")
                 .agg(count("*").alias("count"),
                      mean("avg_mass").alias("mean_mass"),
                      stddev("avg_mass").alias("std_mass"))
               .orderBy("ingest_dt"))
daily_stats.show(100, truncate=False)

# ---------- Percentiles ----------
print("\n=== Percentiles (p50 / p90 / p99) of avg_mass ===")
df.select(
    percentile_approx("avg_mass", 0.5).alias("p50_mass"),
    percentile_approx("avg_mass", 0.9).alias("p90_mass"),
    percentile_approx("avg_mass", 0.99).alias("p99_mass")
).show()

# ---------- Null Rates ----------
print("\n=== Null Rates per Column ===")
null_rates = df.select([
    (count(when(col(c).isNull(), c)) / row_cnt).alias(f"{c}_null_rate")
    for c in df.columns
])
null_rates.show(truncate=False)

# ---------- Stratified Sample ----------
print("\n=== Stratified Sample by Mass Bin (preview) ===")
df_bins = df.withColumn(
    "mass_bin",
    expr("""CASE
              WHEN avg_mass < 100 THEN '<100'
              WHEN avg_mass BETWEEN 100 AND 500 THEN '100-500'
              ELSE '>500'
            END""")
)
stratified = df_bins.sampleBy("mass_bin", {"<100": 0.1, "100-500": 0.1, ">500": 0.1}, seed=42)
stratified.select("drug_id", "drug_name", "avg_mass", "mass_bin", "indication").show(20, truncate=False)

# ---------- Simple Plots ----------
out_dir = os.path.abspath("./plots")
os.makedirs(out_dir, exist_ok=True)

# 1) Distribution of avg_mass (downsample to keep driver memory happy)
mass_sample = (df.select("avg_mass")
                 .where(col("avg_mass").isNotNull())
                 .sample(withReplacement=False, fraction=0.25, seed=7))
mass_pd = mass_sample.toPandas()
plt.figure()
plt.hist(mass_pd["avg_mass"], bins=50)
plt.xlabel("Average Mass")
plt.ylabel("Frequency")
plt.title("Distribution of Drug Avg Mass")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "avg_mass_hist.png"))

# 2) Top 10 indications (handle NULLs + long labels)
top_inds = (df.select(when(col("indication").isNull(), "[NULL]").otherwise(col("indication"))
                    .alias("indication"))
              .groupBy("indication").agg(count("*").alias("drug_count"))
              .orderBy(col("drug_count").desc())
              .limit(10))

top_inds_pd = top_inds.toPandas()
# Ensure strings and shorten very long labels for plot readability
top_inds_pd["indication"] = top_inds_pd["indication"].astype(str).apply(
    lambda s: (s if len(s) <= 80 else s[:77] + "...")
)

plt.figure(figsize=(9,5))
plt.barh(top_inds_pd["indication"], top_inds_pd["drug_count"])
plt.xlabel("Drug Count")
plt.title("Top 10 Drug Indications")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "top_indications.png"))

# 3) Daily ingest counts line plot
daily_pd = daily_stats.toPandas()
plt.figure()
plt.plot(daily_pd["ingest_dt"], daily_pd["count"])
plt.xlabel("ingest_dt")
plt.ylabel("Row Count")
plt.title("Daily Ingest Counts")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "daily_counts.png"))

print("\n✅ EDA completed.")
print(f"   Plots saved to: {out_dir}")
print("   - avg_mass_hist.png")
print("   - top_indications.png")
print("   - daily_counts.png")

spark.stop()
