
import os
import altair as alt
import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession, functions as F

# ---------- Config ----------
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minioStrongPass123"
SILVER_PATH = "s3a://bronze/silver/drugs/"
# -----------------------------

st.set_page_config(page_title="EDA - Drug Dataset", layout="wide")

@st.cache_resource(show_spinner=False)
def get_spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return (
        SparkSession.builder
        .appName("EDA_Streamlit")
        .config("spark.driver.bindAddress","127.0.0.1")
        .config("spark.driver.host","127.0.0.1")
        .config("spark.ui.enabled","false")
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.2.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access","true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled","false")
        .config("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )

# Return the PySpark DF as a RESOURCE (not pickled)
@st.cache_resource(show_spinner=False)
def load_df():
    spark = get_spark()
    return spark.read.format("delta").load(SILVER_PATH).cache()

# Serializable metadata (OK to cache with cache_data)
@st.cache_data(show_spinner=False)
def load_meta():
    df = load_df()
    return int(df.count()), tuple(df.columns)

df = load_df()
row_count, cols = load_meta()

st.title("Exploratory Data Analysis - Drug Dataset")
st.success(f"Connected to Silver Table ✅  • Rows: **{row_count:,}**  • Path: `{SILVER_PATH}`")

with st.expander("Preview (first 20 rows)"):
    st.dataframe(df.limit(20).toPandas())

# -------------------------------------------------------------------------------------
# 1) Null rates per column
# -------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def null_rates():
    total = row_count
    stats = []
    for c in cols:
        nulls = df.filter(F.col(c).isNull()).count()
        stats.append((c, round(100.0 * nulls / total, 2)))
    pdf = pd.DataFrame(stats, columns=["column","null_pct"]).sort_values("null_pct", ascending=False)
    return pdf

null_pdf = null_rates()
bar_nulls = alt.Chart(null_pdf).mark_bar().encode(
    x=alt.X('null_pct:Q', title='Null %'),
    y=alt.Y('column:N', sort='-x', title='Column')
).properties(title="Null Rates per Column")
st.altair_chart(bar_nulls, use_container_width=True)

# -------------------------------------------------------------------------------------
# 2) Avg mass distribution with percentiles (p50 / p90 / p99)
# -------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def mass_distribution_and_percentiles():
    mdf = df.select("avg_mass").where(F.col("avg_mass").isNotNull())
    p50, p90, p99 = mdf.approxQuantile("avg_mass", [0.5, 0.9, 0.99], 0.01)
    pdf = mdf.toPandas()
    return pdf, float(p50), float(p90), float(p99)

mass_pdf, p50, p90, p99 = mass_distribution_and_percentiles()
bins = alt.Bin(maxbins=50)
hist = alt.Chart(mass_pdf).mark_bar().encode(
    x=alt.X('avg_mass:Q', bin=bins, title='Average Mass'),
    y=alt.Y('count()', title='Frequency')
).properties(title="Distribution of Avg Mass")

rules = pd.DataFrame({"p":[p50,p90,p99], "label":["p50","p90","p99"]})
vlines = alt.Chart(rules).mark_rule(strokeDash=[6,4]).encode(
    x='p:Q', color='label:N'
)
st.altair_chart(hist + vlines, use_container_width=True)
st.caption(f"Percentiles: p50={p50:.2f}, p90={p90:.2f}, p99={p99:.2f}")

# -------------------------------------------------------------------------------------
# 3) Daily ingest counts
# -------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def daily_counts():
    dc = (df.groupBy("ingest_dt")
            .agg(F.count("*").alias("row_count"))
            .orderBy("ingest_dt"))
    return dc.toPandas()

if "ingest_dt" in cols:
    daily_pdf = daily_counts()
    line = alt.Chart(daily_pdf).mark_line(point=True).encode(
        x=alt.X('ingest_dt:T', title='Ingest Date'),
        y=alt.Y('row_count:Q', title='Row Count')
    ).properties(title="Daily Ingest Counts")
    st.altair_chart(line, use_container_width=True)
else:
    st.info("Field `ingest_dt` not found; skipping daily counts.")

# -------------------------------------------------------------------------------------
# 4) Mechanism coverage
# -------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def mech_coverage():
    mech = df.select(F.when(F.col("mechanism").isNull(), "Missing Mechanism")
                        .otherwise("With Mechanism").alias("bucket"))
    cov = mech.groupBy("bucket").count().toPandas()
    tot = cov["count"].sum()
    cov["pct"] = cov["count"] / tot
    return cov

if "mechanism" in cols:
    cov_pdf = mech_coverage()
    pie = alt.Chart(cov_pdf).mark_arc().encode(
        theta=alt.Theta(field="pct", type="quantitative"),
        color=alt.Color(field="bucket", type="nominal"),
        tooltip=["bucket","count","pct"]
    ).properties(title="Mechanism Coverage")
    st.altair_chart(pie, use_container_width=True)
else:
    st.info("Field `mechanism` not found; skipping coverage pie.")

# -------------------------------------------------------------------------------------
# 5) Top indications
# -------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def top_indications(k=10):
    ind = (df.withColumn(
                "indication_clean",
                F.when((F.col("indication").isNull()) | (F.col("indication") == "NULL"), F.lit("[NULL]"))
                 .otherwise(F.col("indication")))
           .groupBy("indication_clean").agg(F.count("*").alias("drug_count"))
           .orderBy(F.col("drug_count").desc())
           .limit(k))
    return ind.toPandas()

if "indication" in cols:
    k = st.slider("Top K indications", min_value=5, max_value=20, value=10, step=1)
    top_pdf = top_indications(k)
    barh = alt.Chart(top_pdf).mark_bar().encode(
        x=alt.X('drug_count:Q', title='Drug Count'),
        y=alt.Y('indication_clean:N', sort='-x', title='Indication')
    ).properties(title=f"Top {k} Drug Indications")
    st.altair_chart(barh, use_container_width=True)
else:
    st.info("Field `indication` not found; skipping top indications.")

# -------------------------------------------------------------------------------------
# Insights
# -------------------------------------------------------------------------------------
st.subheader("Auto Insights")
insight_lines = []
null_top = null_pdf.iloc[0]
insight_lines.append(f"Highest null rate: **{null_top['column']}** at **{null_top['null_pct']}%**.")
insight_lines.append(f"Avg mass looks heavy-tailed (p50={p50:.0f}, p90={p90:.0f}, p99={p99:.0f}).")
if "mechanism" in cols:
    miss_row = next((r for _,r in mech_coverage().iterrows() if r['bucket']=="Missing Mechanism"), None)
    if miss_row is not None:
        insight_lines.append(f"Mechanism missing for **{miss_row['pct']*100:.1f}%** of rows.")
if "indication" in cols:
    tp = top_indications(10)
    if len(tp):
        insight_lines.append(f"Most common indication: **{tp.iloc[0]['indication_clean']}** "
                             f"({int(tp.iloc[0]['drug_count'])} records).")
st.write("• " + "\n• ".join(insight_lines))
