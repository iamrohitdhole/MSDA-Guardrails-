#!/usr/bin/env python3
"""
train_text_classifier.py

Baseline text classification using DrugBank samples.

- Input (existing files):
    dq/out/gold/samples/train_sample.csv
    dq/out/gold/samples/val_sample.csv
    dq/out/gold/samples/test_sample.csv

- Features:
    TEXT_COL  = description
    LABEL_COL = indication

- Model:
    TF-IDF + Logistic Regression

- Outputs:
    1) Prints metrics to console:
        - Accuracy
        - Macro Precision / Recall / F1
        - ROC-AUC (macro, if computable)
    2) Saves JSON summary:
        dq/out/gold/model_eval_baseline.json
    3) Saves confusion matrix PNG:
        dq/out/gold/confusion_matrix_baseline.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelBinarizer


# =========================
# Paths & config
# =========================

# project root = one level above scripts/
ROOT = Path(__file__).resolve().parents[1]

# your samples are here: dq/out/gold/samples
SAMPLES_DIR = ROOT / "dq" / "out" / "gold" / "samples"
OUT_DIR = ROOT / "dq" / "out" / "gold"

TRAIN_CSV = SAMPLES_DIR / "train_sample.csv"
VAL_CSV   = SAMPLES_DIR / "val_sample.csv"
TEST_CSV  = SAMPLES_DIR / "test_sample.csv"

TEXT_COL  = "description"   # input text
LABEL_COL = "mechanism"    # target label


# =========================
# Helper functions
# =========================

def load_splits():
    """Load train/val/test CSVs and do basic cleaning."""
    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected file not found: {p}\n"
                f"Make sure your splits exist at dq/out/gold/samples/."
            )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
            raise KeyError(
                f"Missing required columns in {name}_sample.csv. "
                f"Needed TEXT_COL='{TEXT_COL}', LABEL_COL='{LABEL_COL}'. "
                f"Found columns: {df.columns.tolist()}"
            )
        df[TEXT_COL]  = df[TEXT_COL].fillna("")
        df[LABEL_COL] = df[LABEL_COL].fillna("__MISSING__")

    return train_df, val_df, test_df


def train_and_eval(text_train, text_val, text_test,
                   y_train, y_val, y_test,
                   model_name: str):
    """Train TF-IDF + Logistic Regression and compute metrics."""
    print(f"\n=== {model_name} ===")

    # TF-IDF on train+val (no peeking at test)
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(text_train)
    X_val   = tfidf.transform(text_val)   # reserved for future tuning
    X_test  = tfidf.transform(text_test)

    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"Accuracy:          {acc:.3f}")
    print(f"Precision (macro): {prec:.3f}")
    print(f"Recall (macro):    {rec:.3f}")
    print(f"F1-score (macro):  {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": cm,
        "classes": np.unique(y_test),
        "vectorizer": tfidf,
        "classifier": clf,
    }


def save_confusion_matrix(cm, classes, out_path: Path):
    """Save confusion matrix as a PNG heatmap."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix — Baseline (description → indication)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix PNG to: {out_path}")


def compute_roc_auc(clf, tfidf, text_test, y_test):
    """Compute macro ROC-AUC if possible."""
    try:
        X_test = tfidf.transform(text_test)
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        y_score = clf.predict_proba(X_test)

        if y_score.shape[1] == 1:
            auc = roc_auc_score(y_test_bin, y_score)
        else:
            auc = roc_auc_score(
                y_test_bin, y_score, multi_class="ovr", average="macro"
            )
        return float(auc)
    except Exception as e:
        print("Could not compute ROC-AUC:", e)
        return None


# =========================
# Main
# =========================

def main():
    print("Loading data splits from dq/out/gold/samples/ ...")
    train_df, val_df, test_df = load_splits()

    X_train = train_df[TEXT_COL]
    X_val   = val_df[TEXT_COL]
    X_test  = test_df[TEXT_COL]

    y_train = train_df[LABEL_COL]
    y_val   = val_df[LABEL_COL]
    y_test  = test_df[LABEL_COL]

    results = train_and_eval(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        model_name="Baseline - description -> indication",
    )

    # ROC-AUC
    roc_auc = compute_roc_auc(
        results["classifier"],
        results["vectorizer"],
        X_test,
        y_test,
    )
    if roc_auc is not None:
        print(f"ROC-AUC (macro):  {roc_auc:.3f}")
        results["roc_auc_macro"] = roc_auc

    # Save metrics JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_out = OUT_DIR / "model_eval_baseline.json"
    metrics_to_save = {
        "model_name": results["model_name"],
        "accuracy": results["accuracy"],
        "precision_macro": results["precision_macro"],
        "recall_macro": results["recall_macro"],
        "f1_macro": results["f1_macro"],
    }
    if roc_auc is not None:
        metrics_to_save["roc_auc_macro"] = results["roc_auc_macro"]

    with metrics_out.open("w") as f:
        json.dump([metrics_to_save], f, indent=2)
    print(f"\nSaved evaluation summary to: {metrics_out}")

    # Save confusion matrix PNG
    cm_out = OUT_DIR / "confusion_matrix_baseline.png"
    save_confusion_matrix(results["confusion_matrix"], results["classes"], cm_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
