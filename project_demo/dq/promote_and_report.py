#!/usr/bin/env python3
# promote_and_report.py
#
# Promotes "good" rows to the canonical Silver location, archives "bad"
# rows with a timestamped folder, and writes a Markdown DQ report.
#
# Works locally; can also publish the promoted Silver to MinIO/S3 as a Delta table.

import argparse, os, sys, shutil, csv, datetime as dt
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def h(msg): print(f"[PROMOTE] {msg}")

def make_spark(use_s3_delta: bool):
    builder = (SparkSession.builder
               .appName("promote_and_report"))
    if use_s3_delta:
        builder = (builder
            .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
            .config("spark.hadoop.fs.s3a.access.key", "minio")
            .config("spark.hadoop.fs.s3a.secret.key", "minioStrongPass123")
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
    return builder.getOrCreate()

def path_exists(p: str) -> bool:
    return Path(p).exists()

def dir_reset(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

def move_tree(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # If moving across devices, shutil.move handles it; if exists, remove then move
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))

def to_md_table(rows, headers):
    # Simple pipe table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = [("" if v is None else str(v)).replace("\n", " ")[:120] for v in r]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

# replace the whole function
def read_summary_counts(summary_csv_path: Path):
    """
    Accepts either a single CSV file OR a Spark CSV output folder.
    Returns a dict of the last-seen row per 'metric'/'rule'/'name'.
    """
    import csv
    counts = {}
    if not summary_csv_path.exists():
        return counts

    def ingest_file(fp: Path):
        with fp.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row.get("metric") or row.get("rule") or row.get("name") or "row"
                counts[k] = row

    if summary_csv_path.is_dir():
        # typical Spark CSV output: part*.csv plus _SUCCESS
        part_files = sorted(list(summary_csv_path.glob("*.csv"))) or \
                     sorted(list(summary_csv_path.rglob("part-*.csv")))
        for fp in part_files:
            ingest_file(fp)
    else:
        ingest_file(summary_csv_path)

    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--good-path", 
        default="artifacts/dq/good_silver.parquet", 
        help="Path to good output (parquet dir)"
    )
    ap.add_argument(
        "--bad-path",  
        default="artifacts/dq/bad_silver.parquet",  
        help="Path to bad output (parquet dir)"
    )
    ap.add_argument(
        "--summary-csv", 
        default="artifacts/dq/summary.csv", 
        help="CSV with summary metrics (optional)"
    )
    ap.add_argument(
        "--silver-local", 
        default="silver/silver_drugs.parquet", 
        help="Canonical local Silver parquet dir"
    )
    ap.add_argument(
        "--archive-dir",  
        default="artifacts/dq/archive/bad", 
        help="Where to archive bad rows"
    )
    ap.add_argument(
        "--report-md", 
        default="artifacts/dq/dq_report.md", 
        help="Markdown report path to write"
    )
    ap.add_argument(
        "--publish-s3", 
        action="store_true", 
        help="Also publish promoted Silver to MinIO/S3 as Delta"
    )
    ap.add_argument(
        "--s3-delta-path", 
        default="s3a://bronze/silver/drugs_clean/", 
        help="MinIO/S3 Delta target"
    )
    args = ap.parse_args()

    good = Path(args.good_path)
    bad  = Path(args.bad_path)
    if not path_exists(str(good)):
        print(f"[ERROR] Good path not found: {good}", file=sys.stderr); return 2
    if not path_exists(str(bad)):
        print(f"[WARN] Bad path not found: {bad} (continuing)")

    spark = make_spark(use_s3_delta=args.publish_s3)

    # Read good/bad for counts & samples
    df_good = spark.read.parquet(str(good))
    n_good  = df_good.count()

    n_bad = 0
    df_bad = None
    if bad.exists():
        df_bad  = spark.read.parquet(str(bad))
        n_bad   = df_bad.count()

    # Sample a few rows for report
    good_cols = [c for c in df_good.columns if c in ("drug_id","drug_name","avg_mass","ingest_dt")]
    if not good_cols:
        good_cols = df_good.columns[:4]
    good_rows = [tuple(r[c] for c in good_cols) for r in df_good.limit(10).collect()]

    bad_rows = []
    if df_bad is not None and len(df_bad.columns) > 0:
        bad_cols = [c for c in df_bad.columns if c in ("drug_id","drug_name","avg_mass","ingest_dt")]
        if not bad_cols:
            bad_cols = df_bad.columns[:4]
        bad_rows = [tuple(r[c] for c in bad_cols) for r in df_bad.limit(10).collect()]
    else:
        bad_cols = []

    # 1) Promote good -> local Silver
    silver_local = Path(args.silver_local)
    h(f"Promoting GOOD -> {silver_local}")
    dir_reset(silver_local)
    shutil.copytree(good, silver_local)

    # 2) Archive bad with timestamp
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = Path(args.archive_dir)
    archive_dst = archive_root / f"bad_{ts}"
    if bad.exists():
        h(f"Archiving BAD -> {archive_dst}")
        archive_root.mkdir(parents=True, exist_ok=True)
        move_tree(bad, archive_dst)

    # 3) Optionally publish to S3/MinIO as Delta
    s3_target = args.s3_delta_path
    if args.publish_s3:
        h(f"Publishing promoted Silver to Delta @ {s3_target}")
        (df_good.write
            .format("delta")
            .mode("overwrite")
            .save(s3_target))
        h("Delta publish complete.")

    # 4) Build Markdown report
    counts = read_summary_counts(Path(args.summary_csv))
    report = []
    report.append(f"# Silver DQ Promotion Report\n")
    report.append(f"- Timestamp: **{dt.datetime.now().isoformat(timespec='seconds')}**")
    report.append(f"- Good input: `{good}`")
    report.append(f"- Bad input: `{bad}`")
    report.append(f"- Local Silver (promoted): `{silver_local}`")
    if args.publish_s3:
        report.append(f"- S3/MinIO Delta (promoted): `{s3_target}`")
    report.append("")
    report.append("## Row Counts")
    report.append(f"- Good rows: **{n_good}**")
    report.append(f"- Bad rows:  **{n_bad}**")
    if counts:
        report.append("\n### Summary metrics (from summary.csv)\n")
        report.append("*(showing last values per metric)*\n")
        # Make a compact table: metric,key fields,row_count, etc. (best-effort)
        hdr = ["metric"] + [k for k in (next(iter(counts.values())).keys()) if k != "metric"]
        rows = []
        for k, v in counts.items():
            rows.append([k] + [v.get(x,"") for x in hdr[1:]])
        report.append(to_md_table(rows, hdr))
    report.append("\n## Sample GOOD rows")
    report.append(to_md_table(good_rows, good_cols if good_cols else ["c1","c2","c3","c4"]))
    report.append("\n## Sample BAD rows")
    if bad_rows:
        report.append(to_md_table(bad_rows, bad_cols if bad_cols else ["c1","c2","c3","c4"]))
    else:
        report.append("_No bad rows sample available (none or empty dataset)._")

    md_path = Path(args.report_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(report), encoding="utf-8")
    h(f"Report written -> {md_path}")

    print("\n✅ Promotion complete.")
    print(f"   Local Silver : {silver_local}")
    if args.publish_s3:
        print(f"   S3 Delta     : {s3_target}")
    print(f"   Bad archived : {archive_dst if bad.exists() else 'N/A'}")
    print(f"   Report       : {md_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
