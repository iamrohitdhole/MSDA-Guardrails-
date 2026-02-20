# app_streamlit.py
# Streamlit console for Team 07 — Drug Information & Interaction Chatbot with Guardrails
# Uses: pandas, numpy, scikit-learn (TF-IDF), streamlit

import json
import re
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------
# Paths (edit if your tree differs)
# ----------------------------
ROOT = Path(".")
DQ_DIR = ROOT / "dq" / "out"
PLOTS_DIR = ROOT / "plots_useful"
GOLD_DIR = ROOT / "gold"
SAMPLES_DIR = GOLD_DIR / "samples"

PARQUET_PATH = DQ_DIR / "silver_good.parquet"
DQ_SUMMARY_CSV = DQ_DIR / "summary.csv"
SPLIT_MANIFEST = GOLD_DIR / "split_manifest.json"

# Optional: a log file your Kafka consumer writes to (update if you have a different path)
STREAM_LOG = ROOT / "events_pipeline" / "stream.log"

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="DrugBank Chatbot — Data & Demo", layout="wide")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def load_df(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        return pd.DataFrame(columns=[
            "drug_id","drug_name","avg_mass","description","indication","mechanism","toxicity","ingest_dt"
        ])
    df = pd.read_parquet(parquet_path)
    # Coerce types safely
    if "avg_mass" in df.columns:
        df["avg_mass"] = pd.to_numeric(df["avg_mass"], errors="coerce")
    if "ingest_dt" in df.columns:
        df["ingest_dt"] = pd.to_datetime(df["ingest_dt"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_dq_summary(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_manifest(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {}

@st.cache_data(show_spinner=False)
def load_samples(samples_dir: Path):
    out = {}
    for split in ["train_sample.csv", "val_sample.csv", "test_sample.csv"]:
        p = samples_dir / split
        if p.exists():
            out[split.replace("_sample.csv","")] = pd.read_csv(p)
    return out

def number_card(label: str, value, help_text: str = ""):
    col = st.container()
    col.markdown(f"**{label}**")
    col.markdown(f"<h2 style='margin-top:-10px'>{value}</h2>", unsafe_allow_html=True)
    if help_text:
        col.caption(help_text)

def read_stream_status(log_path: Path, tail_lines: int = 50):
    if not log_path.exists():
        return {"exists": False, "last_lines": []}
    lines = log_path.read_text(errors="ignore").splitlines()[-tail_lines:]
    # crude micro-batch detection
    batches = [ln for ln in lines if "micro-batch" in ln.lower() or "wrote" in ln.lower()]
    return {"exists": True, "last_lines": lines, "recent_batches": len(batches)}

# ----------------------------
# Simple Retriever & Verifier
# ----------------------------
class TfidfRetriever:
    def __init__(self, df: pd.DataFrame, fields=("drug_name","indication","mechanism","description")):
        self.df = df.fillna("")
        self.fields = [c for c in fields if c in self.df.columns]
        self.corpus = (self.df[self.fields]
                       .astype(str)
                       .agg(" ".join, axis=1)
                       .tolist())
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        if len(self.corpus) > 0:
            self.tfidf = self.vectorizer.fit_transform(self.corpus)
        else:
            self.tfidf = None

    def search(self, query, topk=10):
        if self.tfidf is None or not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        scores = linear_kernel(q_vec, self.tfidf).flatten()
        top_idx = scores.argsort()[::-1][:topk]
        results = []
        for i in top_idx:
            row = self.df.iloc[i].to_dict()
            row["_score"] = float(scores[i])
            results.append(row)
        return results

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def simple_verifier_score(query: str, row: dict) -> float:
    """Linear comb of lexical features to simulate M2 (interpretable & fast)."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    fields = ["drug_name","indication","mechanism","description"]
    txt = " ".join(str(row.get(c,"")) for c in fields).lower()
    t = set(re.findall(r"[a-z0-9]+", txt))
    jac = jaccard(q, t)

    # field hits
    hits = 0.0
    for f, w in (("drug_name", 0.6), ("indication", 0.3), ("mechanism", 0.2)):
        v = str(row.get(f,"")).lower()
        if any(tok in v for tok in q):
            hits += w

    # length penalty to avoid giant texts dominating
    ln = max(1, len(t))
    len_pen = 1.0 / (1.0 + np.log10(ln))

    # Combine with retriever score if present
    base = row.get("_score", 0.0)
    score = 0.55*base + 0.35*jac + 0.25*hits + 0.10*len_pen
    return max(0.0, min(1.0, score))

# ----------------------------
# Guardrails
# ----------------------------
DANGEROUS_PATTERNS = [
    r"\bdos(e|age)\b", r"\bhow much should i take\b", r"\bprescrib(e|ing|ed)\b",
    r"\boverdose\b", r"\bsuicide\b", r"\bself[-\s]?harm\b", r"\bpii\b",
]
def guardrail_action(query: str, conf: float, min_conf: float = 0.45):
    ql = query.lower()
    if any(re.search(p, ql) for p in DANGEROUS_PATTERNS):
        return "REFUSE", "Medical advice / harm policy", conf
    if conf < min_conf:
        return "WARN", "Low confidence — insufficient evidence", conf
    return "ALLOW", "Safe to answer", conf

# ----------------------------
# Load data
# ----------------------------
df = load_df(PARQUET_PATH)
dq_summary = load_dq_summary(DQ_SUMMARY_CSV)
manifest = load_manifest(SPLIT_MANIFEST)
samples = load_samples(SAMPLES_DIR)
retriever = TfidfRetriever(df)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Team 07 — Demo Console")
st.sidebar.write("DrugBank XML • Delta • Kafka • Splits • Guardrails")
st.sidebar.markdown(f"**Data rows (Silver good):** {len(df):,}")
if manifest:
    st.sidebar.caption(f"Prepared totals: {manifest.get('total', '—')} (70/15/15)")

# ----------------------------
# Tabs
# ----------------------------
tab_dash, tab_explore, tab_plots, tab_retriever, tab_splits, tab_stream = st.tabs(
    ["🏠 Dashboard", "🔎 Explorer", "📈 EDA & Plots", "🎯 Retriever + Guardrails", "🧪 Splits", "📡 Streaming"]
)

# ----------------------------
# Dashboard
# ----------------------------
with tab_dash:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    number_card("Total Parsed (raw)", f"{manifest.get('raw_total','17,430')}", "From DQ summary/logs")
    number_card("Silver Good", f"{len(df):,}", "Passed quality checks")
    prepared_total = manifest.get("total", 18000)
    number_card("Prepared (train+val+test)", f"{prepared_total:,}", "70/15/15 stratified")
    # streaming status
    status = read_stream_status(STREAM_LOG)
    if status["exists"]:
        number_card("Streaming", "Running", f"Recent micro-batches: {status['recent_batches']}")
    else:
        number_card("Streaming", "Not detected", "No stream log found")

    st.divider()
    st.markdown("**Data Quality Summary**")
    if not dq_summary.empty:
        st.dataframe(dq_summary.head(50), use_container_width=True)
    else:
        st.info("No dq/out/summary.csv found. Skipping.")

    st.divider()
    st.markdown("**Split Manifest**")
    if manifest:
        st.json(manifest)
    else:
        st.info("No gold/split_manifest.json found. Run prepare_splits.py first.")

# ----------------------------
# Explorer
# ----------------------------
with tab_explore:
    st.subheader("Data Explorer")
    q = st.text_input("Search by name / indication / mechanism", "")
    colf1, colf2, colf3 = st.columns(3)
    min_m = colf1.number_input("Min avg_mass", value=float(0), step=1.0)
    max_m = colf2.number_input("Max avg_mass", value=float(df["avg_mass"].max() if "avg_mass" in df else 10000), step=1.0)
    only_with_mech = colf3.checkbox("Only rows with mechanism", value=False)

    view = df.copy()
    if q.strip():
        pat = re.compile(re.escape(q), re.I)
        mask = (
            view["drug_name"].astype(str).str.contains(pat) |
            view["indication"].astype(str).str.contains(pat) |
            view["mechanism"].astype(str).str.contains(pat)
        )
        view = view[mask]
    if "avg_mass" in view:
        view = view[(view["avg_mass"] >= min_m) & (view["avg_mass"] <= max_m)]
    if only_with_mech and "mechanism" in view:
        view = view[view["mechanism"].astype(str).str.len() > 0]

    st.write(f"Showing {len(view):,} rows")
    st.dataframe(view.head(100), use_container_width=True)
    st.download_button("Download current view (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="explore_view.csv", mime="text/csv")

# ----------------------------
# EDA & Plots
# ----------------------------
with tab_plots:
    st.subheader("EDA Plots & Data Statistics")
    plot_files = [
        ("daily_ingest_trend.png", "Daily ingest counts — temporal integrity"),
        ("mass_distribution_percentiles.png", "Avg mass distribution with p50/p90/p99"),
        ("mechanism_coverage.png", "Mechanism text coverage"),
        ("null_rates.png", "Null rates per column"),
        ("top_indications.png", "Top indications (imbalance)"),
        ("feature_correlation_heatmap.png", "Correlation among numeric/text length features"),
        ("pipeline_stage_counts.png", "Row counts by pipeline stage"),
        ("split_counts.png", "Prepared dataset split sizes"),
    ]
    cols = st.columns(2)
    for i, (fname, cap) in enumerate(plot_files):
        p = (PLOTS_DIR / fname) if "pipeline_stage" not in fname and "split_counts" not in fname else (ROOT / "charts_36" / fname)
        with cols[i % 2]:
            if p.exists():
                st.image(str(p), caption=cap, use_column_width=True)
            else:
                st.warning(f"Missing: {p}")

# ----------------------------
# Retriever + Guardrails
# ----------------------------
with tab_retriever:
    st.subheader("Search & Safety (M1 + M2 + Guardrails)")
    user_q = st.text_input("Enter a query (e.g., 'warfarin mechanism' or 'pain relief')", "")
    topk = st.slider("Top-K", 1, 20, 5)
    thresh = st.slider("Guardrail min confidence (τ)", 0.0, 1.0, 0.45, 0.05)

    if st.button("Run Search") and user_q.strip():
        hits = retriever.search(user_q, topk=topk)
        if not hits:
            st.info("No results. Try a different query.")
        else:
            rows = []
            for h in hits:
                conf = simple_verifier_score(user_q, h)
                action, reason, conf_out = guardrail_action(user_q, conf, thresh)
                rows.append({
                    "action": action,
                    "confidence": round(conf_out, 3),
                    "reason": reason,
                    "drug_id": h.get("drug_id",""),
                    "drug_name": h.get("drug_name",""),
                    "indication": h.get("indication","")[:160],
                    "mechanism": h.get("mechanism","")[:160],
                    "score_m1": round(h.get("_score",0.0), 3),
                })
            res_df = pd.DataFrame(rows)
            st.dataframe(res_df, use_container_width=True)

            # Compose the top ALLOW card
            allow = res_df[res_df["action"] == "ALLOW"].sort_values("confidence", ascending=False)
            if len(allow):
                best = allow.iloc[0]
                rec = df[df["drug_id"] == best["drug_id"]].iloc[0].to_dict() if "drug_id" in best and str(best["drug_id"]) in df.get("drug_id", pd.Series([])).astype(str).tolist() else hits[0]
                st.success(f"✅ Final Answer (ALLOW) — {rec.get('drug_name','(name)')}")
                st.markdown(f"**Indication:** {rec.get('indication','N/A')}")
                st.markdown(f"**Mechanism:** {rec.get('mechanism','N/A')}")
                st.caption(f"Confidence={best['confidence']}  •  Guardrail τ={thresh}  •  Source=DrugBank (ingest_dt {rec.get('ingest_dt','N/A')})")
            else:
                st.warning("No results passed guardrails. Try lowering τ or refining the query.")

# ----------------------------
# Splits
# ----------------------------
with tab_splits:
    st.subheader("Prepared Dataset Splits (70/15/15)")
    if samples:
        sc1, sc2, sc3 = st.columns(3)
        for name, df_s in [("train", samples.get("train")), ("val", samples.get("val")), ("test", samples.get("test"))]:
            if df_s is None: 
                continue
            with (sc1 if name=="train" else sc2 if name=="val" else sc3):
                st.markdown(f"**{name.title()} Sample**  ({len(df_s):,} rows in sample)")
                st.dataframe(df_s.head(20), use_container_width=True)
                st.download_button(f"Download {name} sample (CSV)", df_s.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{name}_sample.csv", mime="text/csv")
    else:
        st.info("No samples found in gold/samples/. Run prepare_splits.py with --sample-rows.")

# ----------------------------
# Streaming
# ----------------------------
with tab_stream:
    st.subheader("Kafka Streaming Monitor")
    st.caption("Shows last lines from events_pipeline/stream.log (update path if different).")
    status = read_stream_status(STREAM_LOG, tail_lines=80)
    if status["exists"]:
        st.code("\n".join(status["last_lines"]))
        st.metric("Recent micro-batches", value=status["recent_batches"])
    else:
        st.info("No stream log found. Ensure your consumer writes to events_pipeline/stream.log.")
