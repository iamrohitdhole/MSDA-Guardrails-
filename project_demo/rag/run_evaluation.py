#!/usr/bin/env python3
"""
CLI Script to Run Evaluation and Generate Plots

Computes guardrail metrics and generates visualization plots.

Usage:
    python -m project_demo.rag.run_evaluation

Or with options:
    python -m project_demo.rag.run_evaluation --inference-path artifacts/demo2/inference_outputs.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_demo.rag import (
    ARTIFACTS_DIR,
    INFECTION_OUTPUTS_PATH,
    METRICS_PATH,
    PLOTS_DIR,
    GROUNDING_THRESHOLD_DEFAULT,
)
from project_demo.rag.evaluator import run_full_evaluation, compare_demo2_modes

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation and generate plots for RAG guardrail system"
    )
    parser.add_argument(
        "--inference-path",
        type=Path,
        default=INFECTION_OUTPUTS_PATH,
        help="Path to inference outputs JSONL (default: artifacts/demo2/inference_outputs.jsonl)"
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=METRICS_PATH,
        help="Path to save metrics JSON (default: artifacts/demo2/metrics.json)"
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory to save plots (default: artifacts/demo2/plots/)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=GROUNDING_THRESHOLD_DEFAULT,
        help="Grounding threshold for evaluation (default: 0.45)"
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Read artifacts/demo2/inference_<mode>.jsonl and write metrics_by_mode.json + latency_by_mode.png",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Base dir for per-mode inference JSONL (default: artifacts/demo2)",
    )

    args = parser.parse_args()

    # Ensure directories exist
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Plots directory: {args.plots_dir}")
    logger.info(f"Threshold: {args.threshold}")

    if args.compare_modes:
        payload = compare_demo2_modes(
            artifacts_dir=args.artifacts_dir,
            grounding_threshold=args.threshold,
            out_metrics_path=args.artifacts_dir / "metrics_by_mode.json",
            plots_dir=args.plots_dir,
        )
        print("\n✅ Multi-mode evaluation complete!")
        print(f"   Combined metrics: {args.artifacts_dir / 'metrics_by_mode.json'}")
        print(f"   Latency comparison: {args.plots_dir / 'latency_by_mode.png'}")
        print(f"   Modes found: {list(payload.get('modes', {}).keys())}")
        return 0

    # Check inference file exists
    if not args.inference_path.exists():
        logger.error(f"Inference outputs not found: {args.inference_path}")
        logger.error("Run run_inference.py first to generate test data")
        return 1

    logger.info(f"Input: {args.inference_path}")
    logger.info(f"Metrics output: {args.metrics_path}")

    # Run full evaluation
    run_full_evaluation(
        inference_path=args.inference_path,
        metrics_path=args.metrics_path,
        plots_dir=args.plots_dir,
        grounding_threshold=args.threshold,
    )

    # Verify outputs
    print("\n✅ Evaluation complete!")
    print(f"   Metrics: {args.metrics_path}")
    print(f"   Plots:")
    for plot_file in sorted(args.plots_dir.glob("*.png")):
        print(f"      - {plot_file.name} ({plot_file.stat().st_size / 1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
