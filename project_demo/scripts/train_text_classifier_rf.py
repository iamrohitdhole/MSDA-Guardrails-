#!/usr/bin/env python3
"""
train_text_classifier_rf.py

Second model for DrugBank text classification: TF-IDF + Random Forest.
Uses the same data splits and evaluation protocol as the baseline (Logistic Regression)
for side-by-side comparison.

- Input: same as baseline — dq/out/gold/samples/{train,val,test}_sample.csv
- Features: TEXT_COL = description, LABEL_COL = mechanism
- Model: TF-IDF (same params) + RandomForestClassifier
- Outputs: model_eval_rf.json, confusion_matrix_rf.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelBinarizer

# Reuse paths from baseline
ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "dq" / "out" / "gold" / "samples"
OUT_DIR = ROOT / "dq" / "out" / "gold"

TRAIN_CSV = SAMPLES_DIR / "train_sample.csv"
VAL_CSV = SAMPLES_DIR / "val_sample.csv"
TEST_CSV = SAMPLES_DIR / "test_sample.csv"

TEXT_COL = "description"
LABEL_COL = "mechanism"

# RF-specific (keep feature space same as baseline for fair comparison)
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 2


def load_splits():
    """Load train/val/test CSVs; same logic as baseline."""
    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected file not found: {p}\n"
                "Run prepare_splits.py and train_text_classifier.py (baseline) first."
            )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
            raise KeyError(
                f"Missing columns in {name}. Need '{TEXT_COL}', '{LABEL_COL}'. "
                f"Found: {df.columns.tolist()}"
            )
        df[TEXT_COL] = df[TEXT_COL].fillna("")
        df[LABEL_COL] = df[LABEL_COL].fillna("__MISSING__")

    return train_df, val_df, test_df


def train_and_eval(text_train, text_val, text_test, y_train, y_val, y_test):
    """Train TF-IDF + Random Forest and compute same metrics as baseline."""
    print("\n=== Random Forest (description -> mechanism) ===")

    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_train = tfidf.fit_transform(text_train)
    X_val = tfidf.transform(text_val)
    X_test = tfidf.transform(text_test)

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=42,
    )
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
    classes = np.unique(y_test)

    roc_auc = None
    try:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        y_score = clf.predict_proba(X_test)
        if y_score.shape[1] == 1:
            roc_auc = float(roc_auc_score(y_test_bin, y_score))
        else:
            roc_auc = float(
                roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro")
            )
        print(f"ROC-AUC (macro):   {roc_auc:.3f}")
    except Exception as e:
        print("ROC-AUC: not computed:", e)

    return {
        "model_name": "Random Forest - description -> mechanism",
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "roc_auc_macro": roc_auc,
        "confusion_matrix": cm,
        "classes": classes,
        "vectorizer": tfidf,
        "classifier": clf,
    }


def save_confusion_matrix(cm, classes, out_path: Path):
    """Save confusion matrix PNG."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix — Random Forest (description → mechanism)")
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
    print(f"Saved confusion matrix to: {out_path}")


def main():
    print("Loading splits from dq/out/gold/samples/ ...")
    train_df, val_df, test_df = load_splits()

    X_train = train_df[TEXT_COL]
    X_val = val_df[TEXT_COL]
    X_test = test_df[TEXT_COL]
    y_train = train_df[LABEL_COL]
    y_val = val_df[LABEL_COL]
    y_test = test_df[LABEL_COL]

    results = train_and_eval(X_train, X_val, X_test, y_train, y_val, y_test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_to_save = {
        "model_name": results["model_name"],
        "accuracy": results["accuracy"],
        "precision_macro": results["precision_macro"],
        "recall_macro": results["recall_macro"],
        "f1_macro": results["f1_macro"],
    }
    if results.get("roc_auc_macro") is not None:
        metrics_to_save["roc_auc_macro"] = results["roc_auc_macro"]

    metrics_out = OUT_DIR / "model_eval_rf.json"
    with metrics_out.open("w") as f:
        json.dump([metrics_to_save], f, indent=2)
    print(f"\nSaved evaluation summary to: {metrics_out}")

    cm_out = OUT_DIR / "confusion_matrix_rf.png"
    save_confusion_matrix(results["confusion_matrix"], results["classes"], cm_out)

    print("\nDone. Run compare_models.py for side-by-side comparison with baseline.")


if __name__ == "__main__":
    main()
