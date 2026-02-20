from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, percentile_approx, when, expr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

spark = (SparkSession.builder
         .appName("eda_plots_drugs")
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

out_dir = os.path.abspath("./plots_useful")
os.makedirs(out_dir, exist_ok=True)

# 1. Daily ingest trend
daily_stats = df.groupBy("ingest_dt").agg(count("*").alias("count")).orderBy("ingest_dt")
daily_pd = daily_stats.toPandas()
plt.figure()
plt.plot(daily_pd["ingest_dt"], daily_pd["count"], marker="o")
plt.title("Daily Ingest Counts")
plt.xlabel("Ingest Date")
plt.ylabel("Row Count")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "daily_ingest_trend.png"))

# 2. Null rates per column
null_rates = df.select([
    (count(when(col(c).isNull(), c)) / row_cnt).alias(c) for c in df.columns
])
null_pd = null_rates.toPandas().T.reset_index()
null_pd.columns = ["column", "null_rate"]
plt.figure(figsize=(8,5))
plt.bar(null_pd["column"], null_pd["null_rate"]*100)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Null %")
plt.title("Null Rates per Column")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "null_rates.png"))

# 3. Mass distribution with percentiles
percentiles = df.select(
    percentile_approx("avg_mass", 0.5).alias("p50"),
    percentile_approx("avg_mass", 0.9).alias("p90"),
    percentile_approx("avg_mass", 0.99).alias("p99")
).collect()[0].asDict()

mass_pd = df.select("avg_mass").where(col("avg_mass").isNotNull()).sample(0.25, seed=7).toPandas()
plt.figure()
plt.hist(mass_pd["avg_mass"], bins=50, color="skyblue", alpha=0.7)
for p,label in [(percentiles["p50"],"p50"), (percentiles["p90"],"p90"), (percentiles["p99"],"p99")]:
    plt.axvline(p, color="red", linestyle="--", label=f"{label}={round(p,2)}")
plt.legend()
plt.xlabel("Average Mass")
plt.ylabel("Frequency")
plt.title("Distribution of Avg Mass with Percentiles")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mass_distribution_percentiles.png"))

# 4. Mechanism coverage pie chart
coverage = df.select(
    count(when(col("mechanism").isNotNull(), 1)).alias("with_mech"),
    count(when(col("mechanism").isNull(), 1)).alias("missing_mech")
).collect()[0].asDict()
plt.figure()
plt.pie([coverage["with_mech"], coverage["missing_mech"]],
        labels=["With Mechanism", "Missing Mechanism"],
        autopct="%1.1f%%", colors=["green","lightgrey"])
plt.title("Mechanism Coverage")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mechanism_coverage.png"))

# 5. Top indications
top_inds = (df.select(when(col("indication").isNull(),"[NULL]").otherwise(col("indication")).alias("indication"))
              .groupBy("indication").agg(count("*").alias("drug_count"))
              .orderBy(col("drug_count").desc()).limit(10))
top_inds_pd = top_inds.toPandas()
plt.figure(figsize=(9,5))
plt.barh(top_inds_pd["indication"].astype(str), top_inds_pd["drug_count"], color="orange")
plt.gca().invert_yaxis()
plt.xlabel("Drug Count")
plt.title("Top 10 Drug Indications")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "top_indications.png"))

print("\n✅ EDA Plots generated:")
print(" - daily_ingest_trend.png")
print(" - null_rates.png")
print(" - mass_distribution_percentiles.png")
print(" - mechanism_coverage.png")
print(" - top_indications.png")

spark.stop()
