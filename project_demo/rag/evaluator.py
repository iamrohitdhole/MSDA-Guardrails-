"""
Evaluator for RAG Guardrail System

Computes metrics:
- JSVS: JSON Structural Validity Score
- SMR: Schema Match Rate
- FAR: False Accept Rate
- FRR: False Reject Rate
- Precision/Recall vs ground truth
- Confusion Matrix (TP, FP, TN, FN)
- Grounding score distribution
- Latency metrics (mean, p95)
- ROC curve from threshold sweep
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from project_demo.rag import (
    ARTIFACTS_DIR,
    PLOTS_DIR,
    INFECTION_OUTPUTS_PATH,
    METRICS_PATH,
    GROUNDING_THRESHOLD_DEFAULT,
    VALID_MODEL_MODES,
)
from project_demo.rag.demo2_runner import mode_flags
from project_demo.rag.guardrail import (
    check_json_parse,
    check_schema,
    run_guardrail_checks,
)

logger = logging.getLogger(__name__)


# ============================================================
# METRICS DATACLASS
# ============================================================

@dataclass
class GuardrailMetrics:
    """Aggregated metrics from guardrail evaluation."""
    num_samples: int
    jsvs: float  # JSON Structural Validity Score
    smr: float   # Schema Match Rate
    allow_rate: float  # % ALLOW decisions
    block_rate: float  # % BLOCK decisions
    grounding_mean: float
    grounding_std: float
    coverage_mean: float
    grounding_failure_rate: float  # fraction with final rule_tag == LOW_GROUNDING
    far: float  # False Accept Rate
    frr: float  # False Reject Rate
    precision: float
    recall: float
    f1: float
    confusion: Dict[str, int] = field(default_factory=dict)  # TP, FP, TN, FN
    latency_mean_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    optimal_threshold: float = 0.0
    rule_breakdown: Dict[str, int] = field(default_factory=dict)
    model_mode: Optional[str] = None  # set when evaluating a single-mode file


# ============================================================
# CONFUSION MATRIX COMPUTATION
# ============================================================

def compute_confusion_matrix(
    predictions: List[str],  # "ALLOW" or "BLOCK"
    labels: List[bool],      # True = should ALLOW, False = should BLOCK
) -> Dict[str, int]:
    """
    Compute confusion matrix from predictions and ground truth labels.

    Returns dict with:
    - TP: True Positive (predicted ALLOW, labeled ALLOW)
    - FP: False Positive (predicted ALLOW, labeled BLOCK)
    - TN: True Negative (predicted BLOCK, labeled BLOCK)
    - FN: False Negative (predicted BLOCK, labeled ALLOW)
    """
    tp = fp = tn = fn = 0

    for pred, label in zip(predictions, labels):
        if pred == "ALLOW":
            if label:
                tp += 1
            else:
                fp += 1
        else:  # BLOCK
            if label:
                fn += 1
            else:
                tn += 1

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def compute_metrics_from_confusion(confusion: Dict[str, int]) -> Dict[str, float]:
    """
    Compute precision, recall, F1, FAR, FRR from confusion matrix.

    FAR = FP / (FP + TN)  # Invalid accepted / All invalid
    FRR = FN / (TP + FN)  # Valid rejected / All valid
    """
    tp = confusion.get("TP", 0)
    fp = confusion.get("FP", 0)
    tn = confusion.get("TN", 0)
    fn = confusion.get("FN", 0)

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # FAR = FP / (FP + TN)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # FRR = FN / (TP + FN)
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "frr": frr,
    }


# ============================================================
# THRESHOLD SWEEP FOR ROC CURVE
# ============================================================

def sweep_thresholds(
    grounding_scores: List[float],
    labels: List[bool],
    num_thresholds: int = 21,
) -> List[Dict[str, Any]]:
    """
    Sweep grounding threshold from 0.0 to 1.0 and compute metrics at each.

    Returns list of dicts with threshold, far, frr, tpr, fpr.
    """
    results = []

    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    for thresh in thresholds:
        # Predict ALLOW if grounding >= threshold
        predictions = ["ALLOW" if s >= thresh else "BLOCK" for s in grounding_scores]

        confusion = compute_confusion_matrix(predictions, labels)
        metrics = compute_metrics_from_confusion(confusion)

        # TPR = Recall = TP / (TP + FN)
        tpr = metrics["recall"]

        # FPR = FP / (FP + TN) = FAR
        fpr = metrics["far"]

        results.append({
            "threshold": thresh,
            "tpr": tpr,
            "fpr": fpr,
            "far": metrics["far"],
            "frr": metrics["frr"],
            "precision": metrics["precision"],
        })

    return results


def find_optimal_threshold(
    sweep_results: List[Dict[str, Any]],
    method: str = "youden",
) -> float:
    """
    Find optimal threshold using Youden's J statistic.

    Youden's J = TPR - FPR = Sensitivity - Specificity
    Maximize J to find optimal threshold.
    """
    best_j = -float("inf")
    best_thresh = 0.5

    for result in sweep_results:
        j = result["tpr"] - result["fpr"]
        if j > best_j:
            best_j = j
            best_thresh = result["threshold"]

    return best_thresh


# ============================================================
# PLOT GENERATION
# ============================================================

def plot_roc_curve(
    sweep_results: List[Dict[str, Any]],
    output_path: Path,
    optimal_threshold: float,
) -> None:
    """Plot ROC curve from threshold sweep."""
    fpr = [r["fpr"] for r in sweep_results]
    tpr = [r["tpr"] for r in sweep_results]
    thresholds = [r["threshold"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    # Mark optimal threshold point
    opt_idx = np.argmin([abs(r["threshold"] - optimal_threshold) for r in sweep_results])
    ax.plot(fpr[opt_idx], tpr[opt_idx], 'ro', markersize=10,
            label=f'Optimal (τ={optimal_threshold:.2f})')

    ax.set_xlabel('False Positive Rate (FAR)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve - Guardrail Grounding Threshold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotate every 5th sweep point (same index in fpr/tpr/thresholds)
    for j in range(0, len(thresholds), 5):
        ax.annotate(
            f'{thresholds[j]:.2f}',
            (fpr[j], tpr[j]),
            fontsize=8,
            alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"ROC curve saved to {output_path}")


def plot_confusion_matrix(
    confusion: Dict[str, int],
    output_path: Path,
) -> None:
    """Plot confusion matrix heatmap."""
    # Create matrix array
    # [[TN, FP], [FN, TP]]
    matrix = np.array([
        [confusion.get("TN", 0), confusion.get("FP", 0)],
        [confusion.get("FN", 0), confusion.get("TP", 0)],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted BLOCK', 'Predicted ALLOW'])
    ax.set_yticklabels(['Actual BLOCK', 'Actual ALLOW'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if val > matrix.max() / 2 else 'black')

    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Confusion matrix saved to {output_path}")


def plot_grounding_distribution(
    scores: List[float],
    threshold: float,
    output_path: Path,
) -> None:
    """Plot histogram of grounding scores."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(scores, bins=30, edgecolor='black', alpha=0.7)

    # Mark threshold
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold (τ={threshold:.2f})')

    ax.set_xlabel('Grounding Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Grounding Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Grounding distribution saved to {output_path}")


def plot_latency_distribution(
    latencies: List[float],
    output_path: Path,
) -> None:
    """Plot histogram of latency."""
    if not latencies:
        logger.warning("No latency data available")
        # Create placeholder
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No latency data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Latency Distribution')
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)

    ax.hist(latencies, bins=20, edgecolor='black', alpha=0.7)

    ax.axvline(mean_lat, color='green', linestyle='-', linewidth=2,
               label=f'Mean: {mean_lat:.1f}ms')
    ax.axvline(p95_lat, color='orange', linestyle='--', linewidth=2,
               label=f'P95: {p95_lat:.1f}ms')

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Latency distribution saved to {output_path}")


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_inference_outputs(
    inference_path: Path = INFECTION_OUTPUTS_PATH,
    grounding_threshold: float = GROUNDING_THRESHOLD_DEFAULT,
) -> GuardrailMetrics:
    """
    Evaluate inference outputs and compute all metrics.

    Args:
        inference_path: Path to inference_outputs.jsonl
        grounding_threshold: Threshold for grounding check

    Returns:
        GuardrailMetrics dataclass
    """
    # Load inference outputs
    if not inference_path.exists():
        raise FileNotFoundError(f"Inference outputs not found: {inference_path}")

    records = []
    with open(inference_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line")

    logger.info(f"Loaded {len(records)} inference records")

    # Collect results
    predictions = []  # ALLOW/BLOCK
    grounding_scores = []
    coverage_ratios = []
    latencies = []
    rule_counts = {}
    json_valid_count = 0
    schema_valid_count = 0

    # Ground truth: each JSONL line's "should_allow" (from inference; often a name-in-summary heuristic).
    # If prompts used the wrong column (e.g. "Unknown" for every drug), labels collapse and FAR ≈ ALLOW rate.
    labels = []

    inferred_mode: Optional[str] = None
    if records:
        modes_found = {str(r.get("model_mode") or "") for r in records}
        modes_found.discard("")
        if len(modes_found) == 1:
            inferred_mode = next(iter(modes_found))

    low_grounding_final = 0

    for rec in records:
        # Extract fields
        llm_content = rec.get("llm_content", "")
        evidence = rec.get("evidence", [])
        latency_ms = rec.get("latency_ms")
        ground_truth_label = rec.get("should_allow", True)  # Default assume valid

        mm = str(rec.get("model_mode") or inferred_mode or "rag")
        try:
            _, vg, vc, _ = mode_flags(mm)
        except ValueError:
            vg, vc = True, True

        # Run guardrail checks (same enforcement flags as at inference)
        result = run_guardrail_checks(
            llm_content,
            evidence,
            grounding_threshold=grounding_threshold,
            validate_grounding=vg,
            validate_coverage=vc,
        )

        pred = rec.get("decision") or result.decision
        predictions.append(pred)
        gs = rec.get("grounding_score")
        grounding_scores.append(float(gs) if gs is not None else result.grounding_score)
        coverage_ratios.append(
            float(rec["coverage_ratio"]) if rec.get("coverage_ratio") is not None else result.coverage_ratio
        )

        if rec.get("rule_tag") == "LOW_GROUNDING":
            low_grounding_final += 1

        if latency_ms is not None:
            latencies.append(latency_ms)

        # Count rule tags (recomputed decision path)
        tag = rec.get("rule_tag") or result.rule_tag
        if tag:
            rule_counts[tag] = rule_counts.get(tag, 0) + 1

        # Track JSON/Schema validity
        json_valid, _, _ = check_json_parse(llm_content)
        if json_valid:
            json_valid_count += 1

        if json_valid:
            parsed = json.loads(llm_content)
            schema_valid, _ = check_schema(parsed)
            if schema_valid:
                schema_valid_count += 1

        # Ground truth label
        labels.append(ground_truth_label)

    # Compute metrics
    n = len(records)

    jsvs = json_valid_count / n if n > 0 else 0.0
    smr = schema_valid_count / n if n > 0 else 0.0

    allow_count = sum(1 for p in predictions if p == "ALLOW")
    block_count = n - allow_count

    # Confusion matrix
    confusion = compute_confusion_matrix(predictions, labels)
    metric_dict = compute_metrics_from_confusion(confusion)

    # Threshold sweep for ROC
    sweep_results = sweep_thresholds(grounding_scores, labels)
    optimal_threshold = find_optimal_threshold(sweep_results)

    gfr = low_grounding_final / n if n > 0 else 0.0

    # Build metrics
    metrics = GuardrailMetrics(
        num_samples=n,
        jsvs=jsvs,
        smr=smr,
        allow_rate=allow_count / n if n > 0 else 0.0,
        block_rate=block_count / n if n > 0 else 0.0,
        grounding_mean=float(np.mean(grounding_scores)) if grounding_scores else 0.0,
        grounding_std=float(np.std(grounding_scores)) if grounding_scores else 0.0,
        coverage_mean=float(np.mean(coverage_ratios)) if coverage_ratios else 0.0,
        grounding_failure_rate=gfr,
        far=metric_dict["far"],
        frr=metric_dict["frr"],
        precision=metric_dict["precision"],
        recall=metric_dict["recall"],
        f1=metric_dict["f1"],
        confusion=confusion,
        latency_mean_ms=float(np.mean(latencies)) if latencies else None,
        latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
        optimal_threshold=optimal_threshold,
        rule_breakdown=rule_counts,
        model_mode=inferred_mode,
    )

    return metrics


def save_metrics(metrics: GuardrailMetrics, output_path: Path = METRICS_PATH) -> None:
    """Save metrics to JSON file."""
    data = {
        "num_samples": metrics.num_samples,
        "jsvs": metrics.jsvs,
        "smr": metrics.smr,
        "allow_rate": metrics.allow_rate,
        "block_rate": metrics.block_rate,
        "grounding_mean": metrics.grounding_mean,
        "grounding_std": metrics.grounding_std,
        "coverage_mean": metrics.coverage_mean,
        "grounding_failure_rate": metrics.grounding_failure_rate,
        "far": metrics.far,
        "frr": metrics.frr,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "confusion": metrics.confusion,
        "latency_mean_ms": metrics.latency_mean_ms,
        "latency_p95_ms": metrics.latency_p95_ms,
        "optimal_threshold": metrics.optimal_threshold,
        "rule_breakdown": metrics.rule_breakdown,
        "model_mode": metrics.model_mode,
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")


def generate_all_plots(
    metrics: GuardrailMetrics,
    inference_path: Path = INFECTION_OUTPUTS_PATH,
    plots_dir: Path = PLOTS_DIR,
    grounding_threshold: float = GROUNDING_THRESHOLD_DEFAULT,
) -> None:
    """Generate all evaluation plots."""
    # Load data for plots
    records = []
    with open(inference_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    grounding_scores = []
    latencies = []
    labels = []

    for rec in records:
        # Re-run guardrail to get scores (respect mode flags)
        llm_content = rec.get("llm_content", "")
        evidence = rec.get("evidence", [])
        mm = str(rec.get("model_mode") or "rag")
        try:
            _, vg, vc, _ = mode_flags(mm)
        except ValueError:
            vg, vc = True, True

        result = run_guardrail_checks(
            llm_content,
            evidence,
            grounding_threshold=grounding_threshold,
            validate_grounding=vg,
            validate_coverage=vc,
        )
        gs = rec.get("grounding_score")
        grounding_scores.append(float(gs) if gs is not None else result.grounding_score)
        labels.append(rec.get("should_allow", True))

        if rec.get("latency_ms"):
            latencies.append(rec["latency_ms"])

    # Threshold sweep
    sweep_results = sweep_thresholds(grounding_scores, labels)

    # Generate plots (stable filenames for Streamlit / runbook)
    plot_roc_curve(
        sweep_results,
        plots_dir / "roc_curve.png",
        metrics.optimal_threshold,
    )

    plot_confusion_matrix(
        metrics.confusion,
        plots_dir / "confusion_matrix.png",
    )

    plot_grounding_distribution(
        grounding_scores,
        metrics.optimal_threshold,
        plots_dir / "grounding_distribution.png",
    )

    plot_latency_distribution(
        latencies,
        plots_dir / "latency_distribution.png",
    )

    logger.info(f"All plots saved to {plots_dir}")


def run_full_evaluation(
    inference_path: Path = INFECTION_OUTPUTS_PATH,
    metrics_path: Path = METRICS_PATH,
    plots_dir: Path = PLOTS_DIR,
    grounding_threshold: float = GROUNDING_THRESHOLD_DEFAULT,
) -> GuardrailMetrics:
    """
    Run full evaluation pipeline: metrics + plots.

    Returns metrics object.
    """
    logger.info("Running full evaluation...")

    # Compute metrics
    metrics = evaluate_inference_outputs(inference_path, grounding_threshold)

    # Save metrics
    save_metrics(metrics, metrics_path)

    # Generate plots
    generate_all_plots(metrics, inference_path, plots_dir, grounding_threshold)

    # Print summary
    print("\n" + "=" * 50)
    print("GUARDRAIL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Samples evaluated: {metrics.num_samples}")
    print(f"JSVS (JSON valid): {metrics.jsvs:.1%}")
    print(f"SMR (schema valid): {metrics.smr:.1%}")
    print(f"ALLOW rate: {metrics.allow_rate:.1%}")
    print(f"BLOCK rate: {metrics.block_rate:.1%}")
    print(f"Grounding score: {metrics.grounding_mean:.3f} (+/- {metrics.grounding_std:.3f})")
    print(f"Coverage ratio: {metrics.coverage_mean:.3f}")
    print(f"Grounding failure rate (LOW_GROUNDING): {metrics.grounding_failure_rate:.1%}")
    print(f"FAR (False Accept Rate): {metrics.far:.1%}")
    print(f"FRR (False Reject Rate): {metrics.frr:.1%}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"F1 Score: {metrics.f1:.3f}")
    print(f"Optimal threshold: {metrics.optimal_threshold:.2f}")
    if metrics.latency_mean_ms:
        print(f"Latency (mean): {metrics.latency_mean_ms:.1f}ms")
        print(f"Latency (p95): {metrics.latency_p95_ms:.1f}ms")
    print(f"Rule breakdown: {metrics.rule_breakdown}")
    print("=" * 50)

    return metrics


def plot_latency_by_mode_bar(
    mode_to_metrics: Dict[str, GuardrailMetrics],
    output_path: Path,
) -> None:
    """Bar chart: mean and p95 latency per MODEL_MODE (Demo 2 performance comparison)."""
    if not mode_to_metrics:
        return
    modes = list(mode_to_metrics.keys())
    means = [mode_to_metrics[m].latency_mean_ms or 0.0 for m in modes]
    p95s = [mode_to_metrics[m].latency_p95_ms or 0.0 for m in modes]
    x = np.arange(len(modes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, means, w, label="Mean latency (ms)")
    ax.bar(x + w / 2, p95s, w, label="P95 latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by MODEL_MODE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Latency-by-mode plot saved to %s", output_path)


def compare_demo2_modes(
    artifacts_dir: Path = ARTIFACTS_DIR,
    grounding_threshold: float = GROUNDING_THRESHOLD_DEFAULT,
    out_metrics_path: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Aggregate metrics across ``artifacts/demo2/inference_<mode>.jsonl`` files
    (produced by ``run_inference --run-all-modes``).
    """
    out_metrics_path = out_metrics_path or (artifacts_dir / "metrics_by_mode.json")
    plots_dir = plots_dir or PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_mode: Dict[str, Any] = {}
    gm_objects: Dict[str, GuardrailMetrics] = {}

    for mode in VALID_MODEL_MODES:
        p = artifacts_dir / f"inference_{mode}.jsonl"
        if not p.exists():
            logger.warning("Skipping missing inference file: %s", p)
            continue
        m = evaluate_inference_outputs(p, grounding_threshold)
        gm_objects[mode] = m
        per_mode[mode] = {
            "num_samples": m.num_samples,
            "jsvs": m.jsvs,
            "smr": m.smr,
            "far": m.far,
            "frr": m.frr,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "allow_rate": m.allow_rate,
            "block_rate": m.block_rate,
            "latency_mean_ms": m.latency_mean_ms,
            "latency_p95_ms": m.latency_p95_ms,
            "grounding_failure_rate": m.grounding_failure_rate,
            "confusion": m.confusion,
            "rule_breakdown": m.rule_breakdown,
        }

    if gm_objects:
        plot_latency_by_mode_bar(gm_objects, plots_dir / "latency_by_mode.png")

    payload: Dict[str, Any] = {
        "evaluated_at": datetime.utcnow().isoformat(),
        "grounding_threshold": grounding_threshold,
        "modes": per_mode,
    }
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote combined metrics: %s", out_metrics_path)
    return payload
