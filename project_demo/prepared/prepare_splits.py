#!/usr/bin/env python3
"""
prepare_splits.py — Robust Data Preparation (train/val/test)

Features
- Ratios (default 70/15/15), normalized automatically
- Modes: temporal, stratified (with rare-class safety), or random
- Rare-class handling: pool to '__RARE__' and/or send rare rows to train
- Safe stratified split with automatic fallback to random when a class would have <2 rows
- Optional rebalancing so final counts stay near target even when rare→train
- ID overlap (leak) checks
- Outputs: train/val/test (Parquet by default), small *_sample.csv, split_manifest.json

Examples
--------
Stratified, rare-safe (recommended for grading):
  python prepare_splits.py \
    --input dq/out/silver_good.parquet \
    --output-dir dq/out/gold \
    --stratify-col indication \
    --min-per-class 2 \
    --seed 42

Temporal:
  python prepare_splits.py \
    --input dq/out/silver_good.parquet \
    --output-dir dq/out/gold \
    --temporal-col ingest_dt

Stratified with rare→train but keep overall ratios close to target:
  python prepare_splits.py \
    --input dq/out/silver_good.parquet \
    --output-dir dq/out/gold \
    --stratify-col indication \
    --min-per-class 5 \
    --rare-to-train \
    --enforce-target \
    --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None


# -------------------------- IO utils --------------------------
def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if p.suffix.lower() == ".parquet" or p.is_dir():
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported input format: {p}")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Temporal column '{col}' not found.")
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    if out[col].isna().any():
        bad = int(out[col].isna().sum())
        raise ValueError(f"Temporal column '{col}' has {bad} unparsable values.")
    return out


def ratio_tuple(r: Tuple[float, float, float]) -> Tuple[float, float, float]:
    arr = np.array(r, dtype=float)
    if len(arr) != 3:
        raise ValueError("Ratios must be three numbers: train val test")
    if np.any(arr <= 0):
        raise ValueError("All ratios must be > 0")
    arr = arr / arr.sum()
    return float(arr[0]), float(arr[1]), float(arr[2])


# -------------------------- Helpers --------------------------
def _prepare_strat_column_for_split(df: pd.DataFrame, strat_col: str, min_per_class: int) -> pd.DataFrame:
    out = df.copy()
    if out[strat_col].isna().any():
        out[strat_col] = out[strat_col].fillna("__MISSING__")
    counts = out[strat_col].value_counts()
    rare = set(counts[counts < min_per_class].index)
    if rare:
        out[strat_col] = out[strat_col].where(~out[strat_col].isin(rare), "__RARE__")
    return out


def _can_stratify(series: pd.Series) -> bool:
    """Return True if every class has count >= 2 (required by sklearn)."""
    vc = series.value_counts()
    return (vc >= 2).all()


def _safe_train_test_split_stratified(
    df: pd.DataFrame,
    strat_col: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Try a stratified split; if any class is too small, fall back to random split.
    Returns: (df_a, df_b, mode_used) where mode_used in {"stratified", "random_fallback_*"}
    """
    if len(df) == 0:
        return df, df, "skipped_empty"

    if train_test_split is None:
        # No sklearn; must do random
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_b = int(round(len(df) * test_size))
        b_idx = idx[:n_b]
        a_idx = idx[n_b:]
        return df.iloc[a_idx], df.iloc[b_idx], "random_fallback_no_sklearn"

    # If classes are too small, fallback
    if strat_col not in df.columns or not _can_stratify(df[strat_col]):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_b = int(round(len(df) * test_size))
        b_idx = idx[:n_b]
        a_idx = idx[n_b:]
        return df.iloc[a_idx], df.iloc[b_idx], "random_fallback_small_class"

    try:
        a, b = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=df[strat_col],
        )
        return a, b, "stratified"
    except ValueError:
        # Any edge-case error -> random fallback
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_b = int(round(len(df) * test_size))
        b_idx = idx[:n_b]
        a_idx = idx[n_b:]
        return df.iloc[a_idx], df.iloc[b_idx], "random_fallback_exception"


def _rebalance_main_pool_for_rare_to_train(
    n_total: int, n_rare: int, tr: float, va: float, te: float
) -> Tuple[float, float, float]:
    """
    When rare rows are forced to train, adjust the MAIN pool split so that
    train_total ≈ tr * n_total after we add n_rare to train.
    Returns adjusted (tr_main, va_main, te_main) that still sum to 1.
    """
    if n_total <= 0:
        return tr, va, te
    n_main = max(n_total - n_rare, 0)
    if n_main <= 0:
        return 1.0, 0.0, 0.0  # everything rare, all to train

    # target train count overall
    target_train_total = tr * n_total
    # train from main pool needed
    target_train_main = max(target_train_total - n_rare, 0.0)
    tr_main = target_train_main / n_main

    # clamp, keep some room for val/test
    tr_main = float(np.clip(tr_main, 0.01, 0.99))
    rest = 1.0 - tr_main
    if rest < 0.02:
        rest = 0.02
        tr_main = 1.0 - rest

    va_te = va + te
    if va_te <= 0:
        return tr_main, rest, 0.0
    va_main = rest * (va / va_te)
    te_main = rest * (te / va_te)
    # tiny numerical tidy
    s = tr_main + va_main + te_main
    return tr_main / s, va_main / s, te_main / s


# -------------------------- Splitting --------------------------
def split_temporal(df: pd.DataFrame, time_col: str, ratios: Tuple[float, float, float]) -> Dict[str, pd.DataFrame]:
    df = ensure_datetime(df, time_col).sort_values(time_col).reset_index(drop=True)
    n = len(df)
    tr, va, te = ratios
    n_train = int(round(n * tr))
    n_val = int(round(n * va))
    return {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train : n_train + n_val],
        "test": df.iloc[n_train + n_val :],
    }


def split_stratified(
    df: pd.DataFrame,
    strat_col: str,
    ratios: Tuple[float, float, float],
    seed: int,
    min_per_class: int = 2,
    rare_to_train: bool = False,
    enforce_target: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Stratified split in two stages with rare-class handling and safe fallbacks.
    Returns:
      splits: dict of DataFrames
      notes:  dict describing which mode was used at each stage
    """
    if strat_col not in df.columns:
        raise KeyError(f"Stratify column '{strat_col}' not found.")
    df_proc = _prepare_strat_column_for_split(df, strat_col, min_per_class)

    # Separate rare rows to train if requested
    rare_mask = (df_proc[strat_col] == "__RARE__")
    df_rare = df_proc[rare_mask] if rare_to_train else df_proc.iloc[0:0]
    df_main = df_proc[~rare_mask] if rare_to_train else df_proc

    tr, va, te = ratios
    notes: Dict[str, str] = {}

    # Optionally rebalance main pool so final totals land near the target after rare→train
    if rare_to_train and enforce_target:
        tr_main, va_main, te_main = _rebalance_main_pool_for_rare_to_train(
            n_total=len(df_proc), n_rare=len(df_rare), tr=tr, va=va, te=te
        )
        notes["rebalance"] = f"applied (tr_main={tr_main:.3f}, va_main={va_main:.3f}, te_main={te_main:.3f})"
    else:
        tr_main, va_main, te_main = tr, va, te

    # Stage 1: train vs temp (val+test) on MAIN
    test_val_size = 1.0 - tr_main
    df_train_main, df_temp, stage1_mode = _safe_train_test_split_stratified(
        df_main, strat_col, test_val_size, seed
    )
    notes["stage1_mode"] = stage1_mode

    # Stage 2: val vs test inside temp
    if len(df_temp) == 0:
        splits = {"train": pd.concat([df_train_main, df_rare]).reset_index(drop=True),
                  "val": df_temp, "test": df_temp}
        notes["stage2_mode"] = "skipped_empty_temp"
        return splits, notes

    val_size = va_main / (va_main + te_main) if (va_main + te_main) > 0 else 0.5
    df_val, df_test, stage2_mode = _safe_train_test_split_stratified(
        df_temp, strat_col, test_size=1 - val_size, seed=seed
    )
    notes["stage2_mode"] = stage2_mode

    # Merge rare rows into train if requested
    df_train = df_train_main
    if rare_to_train and len(df_rare) > 0:
        df_train = pd.concat([df_train_main, df_rare], axis=0).reset_index(drop=True)
        notes["rare_policy"] = f"rare_to_train (rare={len(df_rare)})"
    else:
        notes["rare_policy"] = "pooled_only" if (df_proc[strat_col] == "__RARE__").any() else "none"

    splits = {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
    return splits, notes


def split_random(df: pd.DataFrame, ratios: Tuple[float, float, float], seed: int) -> Dict[str, pd.DataFrame]:
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    tr, va, te = ratios
    n_train = int(round(n * tr))
    n_val = int(round(n * va))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return {
        "train": df.iloc[train_idx],
        "val": df.iloc[val_idx],
        "test": df.iloc[test_idx],
    }


# -------------------------- Checks & Outputs --------------------------
def check_leakage(splits: Dict[str, pd.DataFrame], id_col: Optional[str]) -> Dict[str, int | str]:
    overlaps = {}
    if id_col and all(id_col in s.columns for s in splits.values()):
        sets = {k: set(v[id_col].astype(str)) for k, v in splits.items()}
        overlaps["train_val_overlap"] = len(sets["train"] & sets["val"])
        overlaps["train_test_overlap"] = len(sets["train"] & sets["test"])
        overlaps["val_test_overlap"] = len(sets["val"] & sets["test"])
    else:
        overlaps["note"] = "ID column missing or not provided; overlap not checked."
    return overlaps


def class_distribution(df: pd.DataFrame, col: str, top_k: int = 20) -> Dict[str, int]:
    return df[col].value_counts().head(top_k).to_dict()


def save_outputs(
    splits: Dict[str, pd.DataFrame],
    outdir: str,
    sample_rows: int = 5,
    as_csv: bool = False,
):
    out = Path(outdir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        data_path = out / f"{name}.{'csv' if as_csv else 'parquet'}"
        if as_csv:
            part.to_csv(data_path, index=False)
        else:
            part.to_parquet(data_path, index=False)
        k = min(sample_rows, len(part))
        part.head(k).to_csv(out / "samples" / f"{name}_sample.csv", index=False)


def build_manifest(
    df_full: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    mode: str,
    ratios: Tuple[float, float, float],
    strat_col: Optional[str],
    time_col: Optional[str],
    id_col: Optional[str],
    overlaps: Dict[str, int | str],
    notes: Optional[Dict[str, str]] = None,
) -> Dict:
    manifest = {
        "mode": mode,
        "total_rows": int(len(df_full)),
        "ratios_target": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "rows": {k: int(len(v)) for k, v in splits.items()},
        "id_column": id_col,
        "overlap_checks": overlaps,
    }
    if notes:
        manifest["notes"] = notes

    if mode == "temporal" and time_col:
        manifest["temporal_column"] = time_col
        manifest["temporal_ranges"] = {
            k: {"min": str(v[time_col].min()), "max": str(v[time_col].max())} for k, v in splits.items()
        }

    if mode in ("stratified",) and strat_col:
        manifest["stratify_column"] = strat_col
        manifest["class_distribution_top"] = {
            split: class_distribution(df, strat_col) for split, df in splits.items()
        }

    return manifest


def write_manifest(manifest: Dict, outdir: str):
    path = Path(outdir) / "split_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# -------------------------- CLI --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create robust train/val/test splits.")
    p.add_argument("--input", required=True, help="Input parquet directory/file or CSV.")
    p.add_argument("--output-dir", required=True, help="Directory to write splits.")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Ratios for train val test.")
    p.add_argument("--temporal-col", default=None, help="Datetime column for temporal split.")
    p.add_argument("--stratify-col", default=None, help="Categorical column for stratified split.")
    p.add_argument("--id-col", default="drug_id", help="Unique ID column for leakage checks.")
    p.add_argument("--min-per-class", type=int, default=2, help="Min rows per class before pooling to '__RARE__'.")
    p.add_argument("--rare-to-train", action="store_true", help="Send rare-class rows straight to train.")
    p.add_argument("--enforce-target", action="store_true",
                   help="Rebalance main pool so final totals match target ratios even with rare→train.")
    p.add_argument("--save-csv", action="store_true", help="Save CSV instead of Parquet.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--sample-rows", type=int, default=5, help="Rows in each *_sample.csv preview.")
    return p.parse_args()


def main():
    args = parse_args()
    ratios = ratio_tuple(tuple(args.ratios))
    df = read_table(args.input)

    if args.temporal_col:
        mode = "temporal"
        splits = split_temporal(df, args.temporal_col, ratios)
        notes = {}
    elif args.stratify_col:
        mode = "stratified"
        splits, notes = split_stratified(
            df,
            args.stratify_col,
            ratios,
            seed=args.seed,
            min_per_class=args.min_per_class,
            rare_to_train=args.rare_to_train,
            enforce_target=args.enforce_target,
        )
    else:
        mode = "random"
        splits = split_random(df, ratios, seed=args.seed)
        notes = {}

    overlaps = check_leakage(splits, args.id_col)
    total = sum(len(v) for v in splits.values())
    realized = {k: (len(v) / total if total else 0.0) for k, v in splits.items()}

    save_outputs(splits, outdir=args.output_dir, sample_rows=args.sample_rows, as_csv=args.save_csv)
    manifest = build_manifest(
        df_full=df,
        splits=splits,
        mode=mode,
        ratios=ratios,
        strat_col=args.stratify_col,
        time_col=args.temporal_col,
        id_col=args.id_col,
        overlaps=overlaps,
        notes=notes,
    )
    manifest["ratios_realized"] = realized
    write_manifest(manifest, args.output_dir)

    print("\n=== Split Summary ===")
    print(f"Mode: {mode}")
    print(f"Target ratios: train={ratios[0]:.3f}, val={ratios[1]:.3f}, test={ratios[2]:.3f}")
    print(f"Realized:      train={realized['train']:.3f}, val={realized['val']:.3f}, test={realized['test']:.3f}")
    if notes:
        for k, v in notes.items():
            print(f"{k}: {v}")
    for k, v in overlaps.items():
        print(f"{k}: {v}")
    print(f"\nWrote splits to: {Path(args.output_dir).resolve()}")
    print(f"Manifest: {Path(args.output_dir) / 'split_manifest.json'}")
    print(f"Samples:  {Path(args.output_dir) / 'samples'}")


if __name__ == "__main__":
    main()
