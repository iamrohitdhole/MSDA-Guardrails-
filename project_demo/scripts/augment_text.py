# project_demo/scripts/augment_text_with_report.py
# - Augments minority classes
# - Writes a clean, screenshot-ready summary to dq/out/augment_summary.txt

import random
from math import ceil
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

# =========================
# Config
# =========================
TARGET_RATIO_OF_MAJORITY = 0.80   # upsample minorities toward 80% of majority size
BOTTOM_N_FALLBACK = 5             # if percentile rule yields none, use bottom-N classes
K_MAX = 3                         # cap per-row replication to avoid blow-ups
TEXT_COLS = ["description", "mechanism", "toxicity"]
SAMPLE_ROWS = 5                   # how many rows to print in the summary
TRUNC = 100                       # truncate long text fields in summary
# =========================

ROOT = Path(__file__).resolve().parents[1]
DELTA_IN   = ROOT / "lake" / "silver" / "silver_good_delta"
DELTA_OUT  = ROOT / "lake" / "feature" / "drugs_augmented_delta"
REPORT_TXT = ROOT / "dq" / "out" / "augment_summary.txt"
REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Spark
# -------------------------
spark = (
    SparkSession.builder
    .appName("AugmentDrugsTextWithReport")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
    .getOrCreate()
)

# Optional: avoid Spark's giant plan truncation warnings during show()
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

# -------------------------
# Load & basic cleanup
# -------------------------
df = spark.read.format("delta").load(str(DELTA_IN))
df = df.filter(F.col("indication").isNotNull() & (F.length("indication") > 0))
for c in TEXT_COLS:
    df = df.withColumn(c, F.lower(F.coalesce(F.col(c), F.lit(""))))

# -------------------------
# Find minority classes
# -------------------------
cnt = df.groupBy("indication").count().cache()

# Use the 25th percentile as a threshold (more inclusive than median)
q25 = cnt.approxQuantile("count", [0.25], 0.0)[0]
minor_q = cnt.filter(F.col("count") <= F.lit(q25)).orderBy("count")
minor_list = [r["indication"] for r in minor_q.collect()]

# Fallback: if none selected, take bottom-N by count
if not minor_list:
    minor_list = [r["indication"] for r in cnt.orderBy("count").limit(BOTTOM_N_FALLBACK).collect()]

# Edge case: if still none (single-class dataset), write original & exit
if not minor_list:
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(str(DELTA_OUT)))
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== AUGMENTATION REPORT ===\n")
        f.write("No eligible minority classes found. Wrote original data as augmented output (no change).\n")
        f.write(f"Output Delta: {DELTA_OUT}\n")
        f.write(f"Original rows: {df.count():,}\n")
    spark.stop()
    raise SystemExit(0)

# -------------------------
# Replication plan toward target
# -------------------------
max_count = cnt.agg(F.max("count").alias("maxc")).first()["maxc"]
target = int(max_count * TARGET_RATIO_OF_MAJORITY)

plan = (
    cnt.withColumn("need", F.greatest(F.lit(0), F.lit(target) - F.col("count")))
       .withColumn("K", F.when(F.col("count") > 0, F.ceil(F.col("need") / F.col("count"))).otherwise(F.lit(0)))
       .withColumn("K", F.when(F.col("K") > K_MAX, F.lit(K_MAX)).otherwise(F.col("K")))
)
plan_minor = plan.filter(F.col("indication").isin(minor_list))

# -------------------------
# Augmentation UDFs
# -------------------------
def rand_delete(tokens, p=0.12):
    if not tokens:
        return tokens
    kept = [t for t in tokens if random.random() > p]
    return kept or tokens

def rand_swap(tokens, n=1):
    toks = tokens[:]
    L = len(toks)
    for _ in range(n):
        if L < 2:
            break
        i, j = random.randrange(L), random.randrange(L)
        toks[i], toks[j] = toks[j], toks[i]
    return toks

def augment_text(text: str, seed: int) -> str:
    random.seed(seed)
    toks = [t for t in (text or "").split() if t]
    if not toks:
        return text or ""
    if random.random() < 0.5:
        toks = rand_delete(toks, p=0.12)
    else:
        toks = rand_swap(toks, n=1)
    return " ".join(toks)

augment_udf = F.udf(lambda s, seed: augment_text(s, int(seed)), T.StringType())

# -------------------------
# Generate augmented rows
# -------------------------
df_with_k = df.join(plan_minor.select("indication", "K"), on="indication", how="left").fillna({"K": 0})
df_minor = df_with_k.filter(F.col("K") > 0)

# Create 'rep' 1..K to replicate rows and generate unique seeds
df_minor = df_minor.withColumn("rep", F.expr("sequence(1, K)")).withColumn("rep", F.explode("rep"))

# Apply augmentation
for c in TEXT_COLS:
    df_minor = df_minor.withColumn(c, augment_udf(F.col(c), F.col("rep")))

aug_df = df_minor.drop("K", "rep")
out_df = df.unionByName(aug_df, allowMissingColumns=True)

# -------------------------
# Write Delta output
# -------------------------
(out_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(str(DELTA_OUT))
)

# -------------------------
# Build a clean, screenshot-ready report
# -------------------------
orig_count = df.count()
aug_count = aug_df.count()
final_count = out_df.count()

# Small helpers to build compact lines
def trunc(s: str, n: int = TRUNC) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")

# Pull a few samples to the driver for clean printing
orig_samples = (
    df.filter(F.col("indication").isin(minor_list))
      .select("drug_id", "indication", "description")
      .limit(SAMPLE_ROWS)
      .toPandas()
      .to_dict("records")
)

aug_samples = (
    aug_df.select("drug_id", "indication", "description")
          .limit(SAMPLE_ROWS)
          .toPandas()
          .to_dict("records")
)

# Minority class list with counts (top few for readability)
minor_counts = (
    plan_minor.select("indication", "count", "K")
              .orderBy("count")
              .limit(max(SAMPLE_ROWS, 10))
              .toPandas()
              .to_dict("records")
)

# Class balance before/after (top few)
pre_bal = (
    cnt.orderBy("count")
       .limit(max(SAMPLE_ROWS, 10))
       .toPandas()
       .to_dict("records")
)
post_bal = (
    out_df.groupBy("indication").count()
          .orderBy("count")
          .limit(max(SAMPLE_ROWS, 10))
          .toPandas()
          .to_dict("records")
)

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("=== AUGMENTATION REPORT ===\n")
    f.write(f"Input Delta:  {DELTA_IN}\n")
    f.write(f"Output Delta: {DELTA_OUT}\n\n")

    f.write(f"Majority class size: {max_count:,}\n")
    f.write(f"Target per minority: {target:,}  (ratio={TARGET_RATIO_OF_MAJORITY:.2f})\n")
    f.write(f"Replication cap K_MAX: {K_MAX}\n")
    f.write(f"Selected minority classes ({len(minor_list)}):\n")
    for mc in minor_counts:
        f.write(f"  - count={mc['count']:>5}, K={mc['K']}: {trunc(mc['indication'], TRUNC)}\n")

    f.write("\n=== COUNTS ===\n")
    f.write(f"Original rows:           {orig_count:,}\n")
    f.write(f"Augmented rows (new):    {aug_count:,}\n")
    f.write(f"Final combined total:    {final_count:,}\n")

    f.write("\n=== SAMPLE ORIGINAL (minority examples) ===\n")
    if orig_samples:
        for r in orig_samples:
            f.write(f"{r['drug_id']}: {trunc(r['indication'])} | {trunc(r['description'])}\n")
    else:
        f.write("(none)\n")

    f.write("\n=== SAMPLE AUGMENTED ===\n")
    if aug_samples:
        for r in aug_samples:
            f.write(f"{r['drug_id']}: {trunc(r['indication'])} | {trunc(r['description'])}\n")
    else:
        f.write("(none)\n")

    f.write("\n=== CLASS BALANCE (before → after) ===\n")
    for i in range(max(len(pre_bal), len(post_bal))):
        pre_line = f"{trunc(pre_bal[i]['indication'])} : {pre_bal[i]['count']}" if i < len(pre_bal) else ""
        post_line = f"{trunc(post_bal[i]['indication'])} : {post_bal[i]['count']}" if i < len(post_bal) else ""
        if pre_line or post_line:
            f.write(f"  pre: {pre_line}\n")
            f.write(f"  post: {post_line}\n")
            f.write("  ---\n")

print(f"\nAugmented Delta written to: {DELTA_OUT}")
print(f"Screenshot-ready summary:   {REPORT_TXT}")

spark.stop()
