#!/usr/bin/env python3
# quality_enforce_silver_fixed.py
#
# Data quality/enforcement for the polished Silver drugs table.
# - Works with local parquet (default) or s3a/Delta path via --input-path
# - Produces good/bad splits + a small summary CSV
#
# Usage (local parquet):
#   spark-submit dq/quality_enforce_silver_fixed.py
#
# Usage (S3/MinIO Delta):
#   spark-submit #     --packages io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 #     dq/quality_enforce_silver_fixed.py #     --input-path s3a://bronze/silver/drugs/
#
# Inputs expected columns from the polished Silver step:
#   drug_id, drug_name, avg_mass, description, indication, mechanism, toxicity, ingest_dt

import argparse
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def make_spark():
    # Spark session for local or MinIO Parquet access
    spark = (
        SparkSession.builder
        .appName("quality_enforce_silver")
        # S3A / MinIO configs
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minio")
        .config("spark.hadoop.fs.s3a.secret.key", "minioStrongPass123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )
    return spark

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", default="project_demo/dq/out/silver_good.parquet",
                    help="Silver input (parquet dir or Delta table path). Default: project_demo/silver/silver_drugs.parquet")
    ap.add_argument("--good-output", default="artifacts/dq/good_silver.parquet",
                    help="Output path for passing records (parquet)")
    ap.add_argument("--bad-output", default="artifacts/dq/bad_silver.parquet",
                    help="Output path for failing records (parquet)")
    ap.add_argument("--summary-csv", default="artifacts/dq/summary.csv",
                    help="Where to write a tiny summary CSV")
    return ap.parse_args()

def main():
    args = parse_args()
    spark = make_spark()

    input_path = args.input_path
    # Try to read as parquet first, then Delta
    try:
        df = spark.read.parquet(input_path)
        fmt = ("parquet")
    except Exception:
        # Fall back to Delta (e.g., s3a path from Silver)
        df = spark.read.format("delta").load(input_path)
        fmt = "delta"

    print(f"[READ] Loaded Silver from {input_path} (fmt={fmt})")
    print("[SCHEMA]")
    df.printSchema()

    required_cols = ["drug_id","drug_name","avg_mass","indication","mechanism","description","toxicity","ingest_dt"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Silver: {missing}")

    total = df.count()

    # ----- Rule-based checks -----
    # 1) Missing/blank names
    r_name_missing = df.filter(F.col("drug_name").isNull() | (F.length(F.trim("drug_name")) == 0))                        .select("drug_id","drug_name") 

    # 2) Missing IDs
    r_id_missing = df.filter(F.col("drug_id").isNull()).select("drug_id","drug_name") 

    # 3) Duplicate IDs
    r_id_dupes = df.groupBy("drug_id").count().filter(F.col("count") > 1).select("drug_id", "count") 

    # 4) ID format rule (DrugBank IDs usually look like DB + digits)
    r_id_badpat = df.filter(~F.col("drug_id").rlike(r"^DB\d+$")).select("drug_id","drug_name") 

    # 5) avg_mass sanity (>0 and < 1e6), nulls fail
    r_mass_invalid = df.filter(
        F.col("avg_mass").isNull() |
        (F.col("avg_mass") <= 0) |
        (F.col("avg_mass") >= 1_000_000)
    ).select("drug_id","drug_name","avg_mass") 

    # 6) Very short text fields (likely junk)
    r_text_short = df.filter(
        (F.length(F.trim("indication")) < 10) | (F.length(F.trim("mechanism")) < 10)
    ).select("drug_id","drug_name","indication","mechanism") 

    # 7) ingest_dt must be present (we don't parse; just non-null/non-empty)
    r_ingest_missing = df.filter(F.col("ingest_dt").isNull() | (F.length(F.trim("ingest_dt")) == 0))                          .select("drug_id","drug_name","ingest_dt") 

    # ----- Outlier detection (IQR) on avg_mass -----
    # Only if there are enough non-nulls
    nn = df.filter(F.col("avg_mass").isNotNull()).count()
    if nn >= 10:
        q1, q3 = df.approxQuantile("avg_mass", [0.25, 0.75], 0.01)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        r_mass_outliers = df.filter((F.col("avg_mass") < lower_iqr) | (F.col("avg_mass") > upper_iqr))                             .select("drug_id","drug_name","avg_mass") 
    else:
        r_mass_outliers = spark.createDataFrame([], schema="drug_id string, drug_name string, avg_mass double") 

    # ----- Collect failing IDs and split -----
    bad_ids = (r_name_missing.select("drug_id")
               .unionByName(r_id_missing.select("drug_id"))
               .unionByName(r_id_dupes.select("drug_id"))
               .unionByName(r_id_badpat.select("drug_id"))
               .unionByName(r_mass_invalid.select("drug_id"))
               .unionByName(r_mass_outliers.select("drug_id"))
               .unionByName(r_text_short.select("drug_id"))
               .unionByName(r_ingest_missing.select("drug_id"))
               .distinct()) 

    df_bad  = df.join(bad_ids, on="drug_id", how="inner")
    df_good = df.join(bad_ids, on="drug_id", how="left_anti") 

    # ----- Write outputs -----
    df_good.write.mode("overwrite").parquet(args.good_output)
    df_bad.write.mode("overwrite").parquet(args.bad_output)

    good_n = df_good.count()
    bad_n  = df_bad.count()

    print("\n[DQ SUMMARY]")
    print(f"Total: {total}")
    print(f"Good : {good_n}")
    print(f"Bad  : {bad_n}")

    # Simple CSV summary
    summary = spark.createDataFrame(
        [(int(total), int(good_n), int(bad_n))],
        "total long, good long, bad long"
    )
    summary.write.mode("overwrite").option("header", True).csv(args.summary_csv)

    # Also emit top issues to console
    def head(df_, title, n=10):
        print(f"\n--- {title} (showing up to {n}) ---")
        try:
            df_.show(n, truncate=False)
        except Exception as e:
            print(f"(could not show: {e})") 

    head(r_name_missing,   "Missing/blank drug_name")
    head(r_id_missing,     "Missing drug_id")
    head(r_id_dupes,       "Duplicate drug_id")
    head(r_id_badpat,      "Bad ID pattern")
    head(r_mass_invalid,   "Invalid avg_mass")
    head(r_mass_outliers,  "IQR outliers avg_mass")
    head(r_text_short,     "Short indication/mechanism")
    head(r_ingest_missing, "Missing ingest_dt") 

    print(f"\n[WRITE] Good  -> {args.good_output}")
    print(f"[WRITE] Bad   -> {args.bad_output}")
    print(f"[WRITE] Summary CSV -> {args.summary_csv}")
    print("\n✅ DQ finished.") 

if __name__ == "__main__":
    sys.exit(main())
