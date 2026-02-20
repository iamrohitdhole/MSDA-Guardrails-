# correlation_heatmap_code.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _ensure_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # avg_mass_norm
    if "avg_mass" in out.columns and "avg_mass_norm" not in out.columns:
        m = pd.to_numeric(out["avg_mass"], errors="coerce")
        out["avg_mass"] = m
        if m.notna().sum() >= 2 and m.std(skipna=True) > 0:
            out["avg_mass_norm"] = (m - m.mean(skipna=True)) / m.std(skipna=True)

    # ingest_dt_ordinal
    if "to_datetime" not in dir(pd):
        pass
    if "ingest_dt" in out.columns and "ingest_dt_ordinal" not in out.columns:
        dt = pd.to_datetime(out["ingest_dt"], errors="coerce")
        out["ingest_dt_ordinal"] = dt.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    # text lengths
    for col in ["description", "indication", "mechanism"]:
        if col in out.columns:
            out[f"{col}_len"] = out[col].astype(str).replace("nan", "").map(len)

    return out

def make_feature_correlation_heatmap(input_path, numeric_cols=None, out_path="feature_correlation_heatmap.png"):
    # Load
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    # Derive helpful numeric columns if needed
    df = _ensure_numeric_features(df)

    # If user didn’t pass cols, choose a sensible default set that exist
    default_pool = [
        "avg_mass", "avg_mass_norm",
        "ingest_dt_ordinal",
        "description_len", "indication_len", "mechanism_len",
    ]
    if numeric_cols is None or len(numeric_cols) == 0:
        cols = [c for c in default_pool if c in df.columns]
    else:
        cols = [c for c in numeric_cols if c in df.columns]

    # Need at least 2 numeric columns
    if len(cols) < 2:
        raise ValueError(f"Need at least 2 numeric columns. Available after derivation: {cols or 'none'}")

    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved heatmap to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a correlation heatmap from a dataset.")
    parser.add_argument("--input", required=True, help="Path to parquet or csv (e.g., ../dq/out/silver_good.parquet)")
    parser.add_argument("--cols", nargs="*", default=[], help="Numeric columns to include (optional)")
    parser.add_argument("--out", default="feature_correlation_heatmap.png", help="Output PNG path")
    args = parser.parse_args()
    make_feature_correlation_heatmap(args.input, args.cols, args.out)
