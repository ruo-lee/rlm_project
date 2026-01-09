"""
Benchmark utility for comparing Baseline vs Optimized RLM.
"""

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termcolor import colored  # noqa: E402

from src.rlm.baseline import BaselineRLMAgent  # noqa: E402
from src.rlm_optimized import RLMAgent  # noqa: E402


def run_benchmark(
    context: str,
    query: str,
    run_baseline: bool = True,
    run_optimized: bool = True,
    output_file: Optional[str] = None,
) -> dict:
    """
    Run benchmark comparing baseline and optimized RLM.

    Args:
        context: The context text to process
        query: The query to answer
        run_baseline: Whether to run baseline version
        run_optimized: Whether to run optimized version
        output_file: Optional file to save results

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "context_length": len(context),
        "baseline": None,
        "optimized": None,
    }

    # Run Baseline
    if run_baseline:
        print(colored("\n" + "=" * 60, "yellow"))
        print(
            colored("  Running BASELINE RLM (Original Paper)", "yellow", attrs=["bold"])
        )
        print(colored("=" * 60, "yellow"))

        baseline_agent = BaselineRLMAgent()
        baseline_start = time.time()
        baseline_answer = baseline_agent.run(context, query)
        baseline_end = time.time()

        results["baseline"] = {
            "answer": baseline_answer[:500],  # Truncate for comparison
            "duration": baseline_end - baseline_start,
            "stats": asdict(baseline_agent.stats),
        }

        print(
            colored(
                f"\n✅ Baseline complete: {baseline_end - baseline_start:.2f}s",
                "yellow",
            )
        )

    # Run Optimized
    if run_optimized:
        print(colored("\n" + "=" * 60, "green"))
        print(
            colored(
                "  Running OPTIMIZED RLM (Our Implementation)", "green", attrs=["bold"]
            )
        )
        print(colored("=" * 60, "green"))

        optimized_agent = RLMAgent()
        optimized_start = time.time()
        optimized_answer = optimized_agent.run(context, query)
        optimized_end = time.time()

        results["optimized"] = {
            "answer": optimized_answer[:500],
            "duration": optimized_end - optimized_start,
            "stats": {
                "total_input_tokens": optimized_agent.stats["total_input_tokens"],
                "total_output_tokens": optimized_agent.stats["total_output_tokens"],
                "total_tokens": optimized_agent.stats["total_tokens"],
                "estimated_cost": optimized_agent.stats["estimated_cost"],
                "llm_calls": optimized_agent.stats.get("llm_calls", 0),
            },
        }

        print(
            colored(
                f"\n✅ Optimized complete: {optimized_end - optimized_start:.2f}s",
                "green",
            )
        )

    # Print comparison
    _print_comparison(results)

    # Save to file
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(colored(f"\n📊 Results saved to: {output_file}", "cyan"))

    return results


def _print_comparison(results: dict) -> None:
    """Print a comparison table of results."""
    print(colored("\n" + "=" * 60, "cyan"))
    print(colored("  📊 BENCHMARK COMPARISON", "cyan", attrs=["bold"]))
    print(colored("=" * 60, "cyan"))

    if results["baseline"] and results["optimized"]:
        b = results["baseline"]
        o = results["optimized"]

        # Calculate improvements
        time_improvement = b["duration"] / o["duration"] if o["duration"] > 0 else 0
        token_improvement = (
            (b["stats"]["total_tokens"] - o["stats"]["total_tokens"])
            / b["stats"]["total_tokens"]
            * 100
            if b["stats"]["total_tokens"] > 0
            else 0
        )
        cost_improvement = (
            (b["stats"]["estimated_cost"] - o["stats"]["estimated_cost"])
            / b["stats"]["estimated_cost"]
            * 100
            if b["stats"]["estimated_cost"] > 0
            else 0
        )

        print(
            f"\n{'Metric':<20} {'Baseline':>15} {'Optimized':>15} {'Improvement':>15}"
        )
        print("-" * 65)
        print(
            f"{'Duration (s)':<20} {b['duration']:>15.2f} {o['duration']:>15.2f} {time_improvement:>14.1f}x"
        )
        print(
            f"{'Total Tokens':<20} {b['stats']['total_tokens']:>15,} {o['stats']['total_tokens']:>15,} {token_improvement:>14.1f}%"
        )
        print(
            f"{'Estimated Cost':<20} ${b['stats']['estimated_cost']:>14.4f} ${o['stats']['estimated_cost']:>14.4f} {cost_improvement:>14.1f}%"
        )
        print(
            f"{'LLM Calls':<20} {b['stats']['llm_calls']:>15} {o['stats']['llm_calls']:>15}"
        )

    elif results["baseline"]:
        b = results["baseline"]
        print("\nBaseline Only:")
        print(f"  Duration: {b['duration']:.2f}s")
        print(f"  Tokens: {b['stats']['total_tokens']:,}")
        print(f"  Cost: ${b['stats']['estimated_cost']:.4f}")

    elif results["optimized"]:
        o = results["optimized"]
        print("\nOptimized Only:")
        print(f"  Duration: {o['duration']:.2f}s")
        print(f"  Tokens: {o['stats']['total_tokens']:,}")
        print(f"  Cost: ${o['stats']['estimated_cost']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLM Benchmark: Baseline vs Optimized")
    parser.add_argument("-q", "--query", type=str, help="Query to test")
    parser.add_argument(
        "-s", "--size", type=str, default="100k", help="Context size (100k, 500k, 1m)"
    )
    parser.add_argument(
        "--baseline-only", action="store_true", help="Run baseline only"
    )
    parser.add_argument(
        "--optimized-only", action="store_true", help="Run optimized only"
    )
    parser.add_argument("-o", "--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Load context
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.local")

    with open("ratings_train.txt", "r") as f:
        full_text = f.read()

    sizes = {"100k": 100000, "500k": 500000, "1m": 1000000}
    context_limit = sizes.get(args.size, 100000)
    context = full_text[:context_limit]

    query = (
        args.query or "이 데이터셋에서 가장 많이 등장하는 긍정적인 단어 3개를 찾아줘."
    )

    run_benchmark(
        context=context,
        query=query,
        run_baseline=not args.optimized_only,
        run_optimized=not args.baseline_only,
        output_file=args.output,
    )
