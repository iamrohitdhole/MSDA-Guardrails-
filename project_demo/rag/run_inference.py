#!/usr/bin/env python3
"""
Batch inference CLI for the drug-information guardrail chatbot.

Usage:
    python -m project_demo.rag.run_inference --mode self_correcting --model mistral
    python -m project_demo.rag.run_inference --mode baseline --model mistral
    python -m project_demo.rag.run_inference --mode rag --model mistral

Run all three modes (same prompts, for evaluation comparison):
    python -m project_demo.rag.run_inference --run-all-modes --model mistral

With explicit input/output paths:
    python -m project_demo.rag.run_inference --mode self_correcting \\
        --input artifacts/demo2/evidence_index.pkl \\
        --output artifacts/demo2/inference_self_correcting.jsonl \\
        --model mistral

Model resolution order: --model > LOCAL_LLM_MODEL env > LLM_MODEL env > config default (mistral)
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_demo.rag import (
    ARTIFACTS_DIR,
    EVIDENCE_INDEX_PATH,
    INFERENCE_OUTPUTS_PATH,
    MODEL_BACKEND,
    MODEL_MODE,
    LOG_EVERY,
    GROUNDING_THRESHOLD_DEFAULT,
    MIN_COVERAGE_RATIO,
    SELF_CORRECT_MAX_PASSES,
    VALID_MODEL_MODES,
    DEFAULT_MODEL,
)
from project_demo.rag.backend import resolve_model
from project_demo.rag.retriever import EvidenceRetriever
from project_demo.rag.demo2_runner import run_demo2_single

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Prompt generation ─────────────────────────────────────────────────────────

def _record_drug_name(drug: Dict[str, Any]) -> str:
    v = drug.get("drug_name") or drug.get("name")
    return str(v).strip() if v and str(v).strip() else "Unknown"


def _record_drug_id(drug: Dict[str, Any]) -> str:
    v = drug.get("drug_id") or drug.get("drugbank_id") or drug.get("id")
    return str(v).strip() if v and str(v).strip() else "unknown"


QUERY_TEMPLATES = [
    ("indication",  "What conditions is {name} used to treat?"),
    ("indication",  "What are the medical uses of {name}?"),
    ("mechanism",   "How does {name} work in the body?"),
    ("mechanism",   "What is the mechanism of action of {name}?"),
    ("safety",      "What are the side effects of {name}?"),
    ("safety",      "What safety warnings are associated with {name}?"),
    ("general",     "Tell me about the drug {name}."),
    ("general",     "What should I know about {name}?"),
    ("interaction", "What drug interactions does {name} have?"),
    ("toxicity",    "What are the toxicity risks of {name}?"),
]


def generate_test_prompts(
    drug_records: List[Dict[str, Any]],
    n_prompts: int = 500,
) -> List[Dict[str, Any]]:
    if not drug_records:
        raise ValueError("drug_records is empty.")
    prompts = []
    for i in range(n_prompts):
        drug = random.choice(drug_records)
        name = _record_drug_name(drug)
        query_type, template = random.choice(QUERY_TEMPLATES)
        prompts.append({
            "prompt_id": i,
            "drug_id": _record_drug_id(drug),
            "drug_name": name,
            "query": template.format(name=name),
            "query_type": query_type,
            "expected_drug": name,
        })
    return prompts


def load_prompts_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load pre-generated prompts from a JSONL file."""
    prompts = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "query" not in obj:
                    obj["query"] = obj.get("prompt", obj.get("text", ""))
                obj.setdefault("prompt_id", i)
                prompts.append(obj)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line %d", i)
    return prompts


def load_drug_records_local(path: str, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load drug records from local parquet (no Spark needed)."""
    import pandas as pd
    import glob as _glob

    p = Path(path)
    if p.is_dir():
        parts = sorted(_glob.glob(str(p / "part-*.parquet")))
        if not parts:
            parts = sorted(_glob.glob(str(p / "*.parquet")))
        df = pd.concat([pd.read_parquet(pp) for pp in parts], ignore_index=True)
    else:
        df = pd.read_parquet(path)

    if sample_limit:
        df = df.head(sample_limit)
    return df.fillna("").to_dict("records")


# ── Inference runner ──────────────────────────────────────────────────────────

def run_mode(
    mode: str,
    prompts: List[Dict[str, Any]],
    retriever: EvidenceRetriever,
    backend: str,
    model_id: str,
    top_k: int,
    grounding_threshold: float,
    coverage_min_ratio: float,
    output_path: Path,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    logger.info("Running mode=%s with model=%s (%d prompts)...", mode, model_id, len(prompts))

    for i, prompt in enumerate(prompts):
        try:
            result = run_demo2_single(
                prompt, retriever,
                backend=backend,
                top_k=top_k,
                llm_model_id=model_id,
                model_mode=mode,
                grounding_threshold=grounding_threshold,
                coverage_min_ratio=coverage_min_ratio,
                self_correct_max_passes=SELF_CORRECT_MAX_PASSES,
            )
        except Exception as e:
            logger.error("Inference error on prompt %d: %s", i, e)
            result = {
                "prompt_id": prompt.get("prompt_id", i),
                "query": prompt.get("query", ""),
                "model_mode": mode,
                "decision": "BLOCK",
                "rule_tag": "RETRIEVAL_FAILURE",
                "grounding_score": 0.0,
                "coverage_ratio": 0.0,
                "latency_ms": 0.0,
                "error": str(e),
            }
        results.append(result)

        if (i + 1) % LOG_EVERY == 0:
            allow_c = sum(1 for r in results if r.get("decision") == "ALLOW")
            logger.info("[%s] %d/%d  ALLOW=%d", mode, i + 1, len(prompts), allow_c)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    allow_c = sum(1 for r in results if r.get("decision") == "ALLOW")
    lats = [r.get("latency_ms", 0) for r in results if r.get("latency_ms")]
    lat_mean = float(np.mean(lats)) if lats else 0.0
    lat_p95 = float(np.percentile(lats, 95)) if lats else 0.0
    print(
        f"\nmode={mode}  ALLOW={allow_c}/{len(results)}"
        f"  mean_lat={lat_mean:.0f}ms  p95={lat_p95:.0f}ms"
        f"  → {output_path}"
    )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch inference for the drug-information guardrail chatbot."
    )

    # Core args
    parser.add_argument(
        "--mode", "--model-mode",
        dest="mode",
        default=None,
        choices=list(VALID_MODEL_MODES),
        help="Inference mode: baseline | rag | self_correcting (default: MODEL_MODE env or self_correcting)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model tag (default: LOCAL_LLM_MODEL env or {DEFAULT_MODEL}). "
             f"Examples: mistral, llama3, llama3.1, gemma2, phi3, mixtral, llama3.2",
    )
    parser.add_argument(
        "--run-all-modes",
        action="store_true",
        help="Run baseline + rag + self_correcting on the same prompts (for evaluation comparison).",
    )

    # I/O
    parser.add_argument(
        "--input", "--index-path",
        dest="index_path",
        type=Path,
        default=EVIDENCE_INDEX_PATH,
        help=f"Evidence index path. Default: {EVIDENCE_INDEX_PATH}",
    )
    parser.add_argument(
        "--output", "--output-path",
        dest="output_path",
        type=Path,
        default=INFERENCE_OUTPUTS_PATH,
        help=f"Output JSONL path. Default: {INFERENCE_OUTPUTS_PATH}",
    )
    parser.add_argument(
        "--input-prompts",
        type=Path,
        default=None,
        help="Pre-generated prompts JSONL (one per line with 'query' field). "
             "If not given, prompts are auto-generated from drug records.",
    )

    # Prompt generation
    parser.add_argument(
        "--n-prompts", type=int, default=100,
        help="Number of test prompts to generate (default: 100).",
    )
    parser.add_argument(
        "--from-parquet",
        metavar="PATH",
        default="silver/silver_drugs_from_delta.parquet",
        help="Local parquet path for drug records (prompt generation). "
             "Default: silver/silver_drugs_from_delta.parquet",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=None,
        help="Limit drug records loaded for prompt generation.",
    )

    # Guardrail config
    parser.add_argument("--top-k", type=int, default=5, help="Evidence snippets to retrieve (default: 5).")
    parser.add_argument(
        "--grounding-threshold", type=float, default=GROUNDING_THRESHOLD_DEFAULT,
        help=f"Grounding threshold τ (default: {GROUNDING_THRESHOLD_DEFAULT}).",
    )
    parser.add_argument(
        "--coverage-min-ratio", type=float, default=MIN_COVERAGE_RATIO,
        help=f"Minimum coverage ratio (default: {MIN_COVERAGE_RATIO}).",
    )

    # Misc
    parser.add_argument("--backend", default=None, choices=["existing_llm", "local_openai"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # Resolve model
    model_id = resolve_model(args.model)
    if not model_id:
        logger.error(
            "No model specified. Use --model or set LOCAL_LLM_MODEL env var.\n"
            "  Example: --model mistral\n"
            "  Or:      export LOCAL_LLM_MODEL=mistral"
        )
        return 1

    backend = args.backend or MODEL_BACKEND
    mode = args.mode or MODEL_MODE

    logger.info("model=%s  backend=%s  mode=%s", model_id, backend, mode)

    # Load index
    if not args.index_path.exists():
        logger.error(
            "Evidence index not found: %s\n"
            "Build it first:\n"
            "  python -m project_demo.rag.build_index --from-parquet silver/silver_drugs_from_delta.parquet",
            args.index_path,
        )
        return 1

    logger.info("Loading evidence index from %s...", args.index_path)
    retriever = EvidenceRetriever()
    retriever.load_index(args.index_path)

    # Load prompts
    if args.input_prompts and args.input_prompts.exists():
        logger.info("Loading prompts from %s", args.input_prompts)
        prompts = load_prompts_from_jsonl(args.input_prompts)
    else:
        logger.info("Generating %d prompts from drug records...", args.n_prompts)
        try:
            records = load_drug_records_local(args.from_parquet, sample_limit=args.sample_limit)
        except Exception as e:
            logger.warning("Could not load parquet (%s); using minimal fallback records.", e)
            records = [
                {"drug_id": str(i), "drug_name": f"Drug {i}",
                 "description": "Test drug", "indication": "Test indication"}
                for i in range(50)
            ]
        prompts = generate_test_prompts(records, n_prompts=args.n_prompts)

    if not prompts:
        logger.error("No prompts to run.")
        return 1

    logger.info("Running inference on %d prompts...", len(prompts))

    # Run modes
    if args.run_all_modes:
        for m in VALID_MODEL_MODES:
            out = ARTIFACTS_DIR / f"inference_{m}.jsonl"
            run_mode(
                m, prompts, retriever, backend, model_id,
                args.top_k, args.grounding_threshold, args.coverage_min_ratio, out,
            )
        print("\n" + "=" * 60)
        print("ALL MODES COMPLETE")
        print("=" * 60)
        print(f"Outputs: {ARTIFACTS_DIR}/inference_<mode>.jsonl")
        print("Next: python -m project_demo.rag.run_evaluation --compare-modes")
    else:
        run_mode(
            mode, prompts, retriever, backend, model_id,
            args.top_k, args.grounding_threshold, args.coverage_min_ratio, args.output_path,
        )
        print("\n" + "=" * 60)
        print("INFERENCE COMPLETE")
        print("=" * 60)
        print(f"Output: {args.output_path}")
        print("Next: python -m project_demo.rag.run_evaluation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
