#!/usr/bin/env python3
"""
compare_models.py

Side-by-side comparison of Baseline (TF-IDF + Logistic Regression) and
Random Forest model for DrugBank text classification (description -> mechanism).

Reads: dq/out/gold/model_eval_baseline.json, dq/out/gold/model_eval_rf.json
Writes: dq/out/gold/model_comparison.json, dq/out/gold/model_comparison.txt (human-readable)
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dq" / "out" / "gold"

BASELINE_JSON = OUT_DIR / "model_eval_baseline.json"
RF_JSON = OUT_DIR / "model_eval_rf.json"

METRICS_KEYS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]


def load_metrics(path: Path) -> dict:
    with path.open() as f:
        raw = f.read()
    # Handle NaN in JSON (baseline may have written "NaN" as string)
    raw = raw.replace('NaN', 'null')
    data = json.loads(raw)
    if isinstance(data, list) and data:
        return data[0]
    return data


def compare():
    if not BASELINE_JSON.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found: {BASELINE_JSON}\n"
            "Run: python -m project_demo.scripts.train_text_classifier"
        )
    if not RF_JSON.exists():
        raise FileNotFoundError(
            f"RF metrics not found: {RF_JSON}\n"
            "Run: python -m project_demo.scripts.train_text_classifier_rf"
        )

    baseline = load_metrics(BASELINE_JSON)
    rf = load_metrics(RF_JSON)

    # Build side-by-side table
    table = []
    for key in METRICS_KEYS:
        vb = baseline.get(key)
        vr = rf.get(key)
        if vb is None or (isinstance(vb, float) and str(vb) == "nan"):
            vb = None
        if vr is None or (isinstance(vr, float) and str(vr) == "nan"):
            vr = None
        diff = None
        if isinstance(vb, (int, float)) and isinstance(vr, (int, float)):
            diff = round(vr - vb, 4)
        table.append({
            "metric": key,
            "baseline": vb,
            "random_forest": vr,
            "difference_rf_vs_baseline": diff,
        })

    comparison = {
        "baseline_model": baseline.get("model_name", "Baseline (TF-IDF + Logistic Regression)"),
        "new_model": rf.get("model_name", "Random Forest (TF-IDF + RF)"),
        "metrics_side_by_side": table,
        "what_changed": summarize_changes(baseline, rf, table),
    }

    # Save JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "model_comparison.json"
    with out_json.open("w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {out_json}")

    # Human-readable report
    out_txt = OUT_DIR / "model_comparison.txt"
    lines = [
        "=" * 60,
        "MODEL COMPARISON: Baseline vs Random Forest",
        "Task: description -> mechanism (DrugBank)",
        "=" * 60,
        "",
        "Side-by-side metrics",
        "-" * 40,
    ]
    for row in table:
        m, vb, vr, diff = row["metric"], row["baseline"], row["random_forest"], row["difference_rf_vs_baseline"]
        vb_s = f"{vb:.4f}" if isinstance(vb, (int, float)) else str(vb)
        vr_s = f"{vr:.4f}" if isinstance(vr, (int, float)) else str(vr)
        diff_s = f" (Δ = {diff:+.4f})" if diff is not None else ""
        lines.append(f"  {m:20s}  Baseline: {vb_s:8s}  RF: {vr_s:8s}{diff_s}")
    lines.extend([
        "",
        "What changed",
        "-" * 40,
        comparison["what_changed"],
        "",
        "=" * 60,
    ])
    with out_txt.open("w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_txt}")

    # Print to console
    print("\n" + "\n".join(lines))
    return comparison


def summarize_changes(baseline: dict, rf: dict, table: list) -> str:
    parts = []
    best_model_acc = None
    best_model_f1 = None
    for row in table:
        m, vb, vr, diff = row["metric"], row["baseline"], row["random_forest"], row["difference_rf_vs_baseline"]
        if diff is None:
            continue
        if m == "accuracy":
            best_model_acc = "Random Forest" if diff > 0 else "Baseline"
        if m == "f1_macro":
            best_model_f1 = "Random Forest" if diff > 0 else "Baseline"

    if best_model_acc:
        parts.append(f"- Accuracy: {best_model_acc} performs better.")
    if best_model_f1:
        parts.append(f"- F1 (macro): {best_model_f1} performs better.")

    improvements = [r for r in table if r.get("difference_rf_vs_baseline") is not None and r["difference_rf_vs_baseline"] > 0]
    declines = [r for r in table if r.get("difference_rf_vs_baseline") is not None and r["difference_rf_vs_baseline"] < 0]
    if improvements:
        parts.append(f"- Metrics improved with RF: {', '.join(r['metric'] for r in improvements)}.")
    if declines:
        parts.append(f"- Metrics lower with RF: {', '.join(r['metric'] for r in declines)}.")

    parts.append(
        "Baseline uses TF-IDF + Logistic Regression (fast, interpretable). "
        "Random Forest uses same TF-IDF features with an ensemble of trees (can capture non-linear patterns; may generalize differently)."
    )
    return " ".join(parts) if parts else "No numeric comparison available."


if __name__ == "__main__":
    compare()
