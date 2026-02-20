# tokenize_text.py
from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from spark_builder import build_spark
from config_io import (
    SILVER_DELTA_LOCAL, SILVER_DELTA_S3,
    TOK_DELTA_LOCAL,   TOK_DELTA_S3,
)
from io_utils import choose_read_uri, write_delta_both

spark = build_spark("TokenizeText")
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

SILVER = choose_read_uri(str(SILVER_DELTA_LOCAL), SILVER_DELTA_S3)
df = spark.read.format("delta").load(SILVER)

text_cols = ["description","mechanism","toxicity"]
for c in text_cols:
    df = df.withColumn(c, F.lower(F.coalesce(F.col(c), F.lit(""))))

def tokenize(col):
    tok = RegexTokenizer(inputCol=col, outputCol=f"{col}_tok", pattern="\\W+")
    sw = StopWordsRemover(inputCol=f"{col}_tok", outputCol=f"{col}_tokens")
    return tok, sw

for c in text_cols:
    t, s = tokenize(c)
    df = t.transform(df)
    df = s.transform(df)

# Optional TF-IDF (comment out if not needed)
def tfidf(df, col, num_features=65536):
    tf = HashingTF(inputCol=f"{col}_tokens", outputCol=f"{col}_tf", numFeatures=num_features)
    df2 = tf.transform(df)
    idf = IDF(inputCol=f"{col}_tf", outputCol=f"{col}_tfidf")
    model = idf.fit(df2)
    return model.transform(df2)

for c in text_cols:
    df = tfidf(df, c)

out = df.select(
    "drug_id","drug_name","avg_mass","indication","ingest_dt",
    "description_tokens","mechanism_tokens","toxicity_tokens",
    "description_tfidf","mechanism_tfidf","toxicity_tfidf"
)

# Write to BOTH
write_delta_both(out, str(TOK_DELTA_LOCAL), TOK_DELTA_S3)

print("\n=== SAMPLE: Tokenized columns ===")
out.select("drug_id","description_tokens","mechanism_tokens","toxicity_tokens").show(3, truncate=False)

print("\n=== TOKEN STATS ===")
out.select(
    F.size("description_tokens").alias("desc_tok_len"),
    F.size("mechanism_tokens").alias("mech_tok_len"),
    F.size("toxicity_tokens").alias("tox_tok_len")
).agg(
    F.mean("desc_tok_len").alias("avg_desc_len"),
    F.mean("mech_tok_len").alias("avg_mech_len"),
    F.mean("tox_tok_len").alias("avg_tox_len")
).show(truncate=False)

spark.stop()
