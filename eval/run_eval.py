"""
RAGAS evaluation runner.

Measures retrieval and generation quality with four metrics:
- context_precision: Are the top retrieved chunks actually relevant?
- context_recall: Does the context contain all necessary information?
- faithfulness: Does the answer stick to what the context says?
- answer_relevancy: Does the answer address the actual question?

Runs a side-by-side comparison of Groq (Llama 3.3 70B) vs OpenAI (GPT-4o-mini).

Usage:
    python eval/run_eval.py --repo-id <repo_id> --provider groq
    python eval/run_eval.py --repo-id <repo_id> --compare  # Both providers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("eval")

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {DATASET_PATH}. "
            "Create eval/dataset.json with ground-truth Q&A pairs."
        )
    with open(DATASET_PATH) as f:
        return json.load(f)


def run_pipeline(
    question: str,
    repo_id: str,
    provider: str,
    model: str | None = None,
) -> dict:
    """Run the full retrieve + generate pipeline for a single question."""
    from src.retrieval.retriever import retrieve_chunks
    from src.retrieval.generator import get_generator
    from src.config import get_settings

    settings = get_settings()
    effective_model = model or (
        settings.llm_model if provider == "groq" else "gpt-4o-mini"
    )

    # Retrieve
    chunks = retrieve_chunks(question=question, repo_id=repo_id, top_k=8)
    contexts = [c.content for c in chunks]

    # Generate
    generator = get_generator()
    answer, model_used, usage = generator.generate(
        question=question,
        chunks=chunks,
        provider=provider,
        model=effective_model,
    )

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "model": model_used,
        "usage": usage,
    }


def run_ragas_eval(results: list[dict], ground_truths: list[str]) -> dict:
    """
    Run RAGAS evaluation on a set of pipeline results.
    Returns a dict of metric_name → average_score.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        )
    except ImportError as e:
        logger.error(f"RAGAS dependencies not installed: {e}")
        return {}

    eval_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(eval_data)

    logger.info("Running RAGAS evaluation...")
    eval_result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    scores = eval_result.to_pandas().mean(numeric_only=True).to_dict()
    return scores


def print_results_table(provider: str, scores: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"RAGAS Results — {provider}")
    print(f"{'=' * 50}")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<25} {score:.4f}  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--repo-id", required=True, help="Repo ID to evaluate against")
    parser.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "openai"],
        help="LLM provider to evaluate",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare both Groq and OpenAI side by side",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model (e.g., gpt-4o-mini, llama-3.3-70b-versatile)",
    )
    args = parser.parse_args()

    dataset = _load_dataset()
    logger.info(f"Loaded {len(dataset)} evaluation examples")

    questions = [item["question"] for item in dataset]
    ground_truths = [item["ground_truth"] for item in dataset]

    providers = ["groq", "openai"] if args.compare else [args.provider]
    all_scores: dict[str, dict] = {}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for provider in providers:
        logger.info(f"\nRunning pipeline with provider={provider}...")
        results = []
        for i, question in enumerate(questions):
            logger.info(f"  [{i+1}/{len(questions)}] {question[:80]}...")
            try:
                result = run_pipeline(
                    question=question,
                    repo_id=args.repo_id,
                    provider=provider,
                    model=args.model,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"  Failed: {e}")
                results.append({
                    "question": question,
                    "answer": "ERROR",
                    "contexts": [],
                    "model": f"{provider}/error",
                    "usage": {},
                })

        # Run RAGAS
        scores = run_ragas_eval(results, ground_truths)
        all_scores[provider] = scores
        print_results_table(provider, scores)

        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = RESULTS_DIR / f"eval_{provider}_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump({
                "provider": provider,
                "repo_id": args.repo_id,
                "timestamp": timestamp,
                "scores": scores,
                "results": results,
            }, f, indent=2)
        logger.info(f"Results saved to {out_file}")

    # If comparing, print diff table
    if args.compare and len(all_scores) == 2:
        print("\n" + "=" * 60)
        print("Comparison: Groq vs OpenAI")
        print("=" * 60)
        groq_s = all_scores.get("groq", {})
        oai_s = all_scores.get("openai", {})
        metrics = set(list(groq_s.keys()) + list(oai_s.keys()))
        print(f"  {'Metric':<25} {'Groq':>8}  {'OpenAI':>8}  {'Δ':>8}")
        print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*8}")
        for metric in sorted(metrics):
            g = groq_s.get(metric, 0.0)
            o = oai_s.get(metric, 0.0)
            diff = o - g
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            print(f"  {metric:<25} {g:>8.4f}  {o:>8.4f}  {diff_str:>8}")


if __name__ == "__main__":
    main()
