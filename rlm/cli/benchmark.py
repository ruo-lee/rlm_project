#!/usr/bin/env python3
"""
RLM Benchmark CLI - Compare models or baseline vs optimized.

Usage:
    python -m rlm.cli.benchmark --project 1 --query "질문" --models "gemini-3-flash-preview,gemini-2.5-flash"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from termcolor import colored

load_dotenv(dotenv_path=".env.local")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RLM Benchmark - Compare models on the same query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="1",
        help="Project number (run with --list to see available)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Query to benchmark",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="gemini-3-flash-preview,gemini-2.5-flash",
        help="Comma-separated list of models to compare",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available projects",
    )

    return parser.parse_args()


def list_projects():
    """List available projects."""
    from rlm.data import get_all_datasets

    print(colored("\n📂 Available Projects:", "cyan", attrs=["bold"]))
    print("-" * 50)

    for key, info in get_all_datasets().items():
        status = "✅" if os.path.exists(info["path"]) else "❌"
        print(f"  [{key}] {info['name']} {status}")
    print()


def run_model_benchmark(
    project_path: str, query: str, models: list[str], output_file: Optional[str] = None
):
    """Run benchmark comparing multiple models."""
    from rlm.core import RLMAgent

    results = {
        "timestamp": datetime.now().isoformat(),
        "project": project_path,
        "query": query,
        "models": {},
    }

    for model_name in models:
        print(colored(f"\n{'='*60}", "cyan"))
        print(colored(f"  Running: {model_name}", "cyan", attrs=["bold"]))
        print(colored(f"{'='*60}", "cyan"))

        try:
            agent = RLMAgent(model_name=model_name)
            start_time = time.time()
            answer = agent.run(project_path, query)
            end_time = time.time()

            results["models"][model_name] = {
                "answer": answer[:500] if answer else "",
                "duration": end_time - start_time,
                "total_tokens": agent.stats.get("total_tokens", 0),
                "estimated_cost": agent.stats.get("estimated_cost", 0),
                "llm_calls": agent.stats.get("llm_calls", 0),
            }

            print(colored(f"✅ Complete: {end_time - start_time:.2f}s", "green"))

        except Exception as e:
            print(colored(f"❌ Error: {e}", "red"))
            results["models"][model_name] = {"error": str(e)}

    # Print comparison
    print_comparison(results)

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(colored(f"\n📊 Results saved to: {output_file}", "cyan"))

    return results


def print_comparison(results: dict):
    """Print comparison table."""
    print(colored(f"\n{'='*80}", "cyan"))
    print(colored("  📊 BENCHMARK COMPARISON", "cyan", attrs=["bold"]))
    print(colored(f"{'='*80}", "cyan"))

    models = results.get("models", {})
    if not models:
        print("No results to compare.")
        return

    # Header
    print(f"\n{'Model':<30} {'Duration':>12} {'Tokens':>12} {'Cost':>12} {'Calls':>8}")
    print("-" * 78)

    # Data rows
    for model_name, data in models.items():
        if "error" in data:
            print(f"{model_name:<30} {'ERROR':>12}")
        else:
            print(
                f"{model_name:<30} {data['duration']:>11.2f}s {data['total_tokens']:>12,} ${data['estimated_cost']:>11.4f} {data['llm_calls']:>8}"
            )

    print("-" * 78)


def main():
    args = parse_args()

    if args.list:
        list_projects()
        return

    # Get project path
    from rlm.data import get_all_datasets

    all_datasets = get_all_datasets()

    if args.project not in all_datasets:
        print(
            colored(
                f"❌ Project '{args.project}' not found. Use --list to see available.",
                "red",
            )
        )
        sys.exit(1)

    project_path = all_datasets[args.project]["path"]
    models = [m.strip() for m in args.models.split(",")]

    print(colored("\n🚀 RLM Benchmark", "cyan", attrs=["bold"]))
    print(f"   Project: {project_path}")
    print(f"   Query: {args.query[:50]}...")
    print(f"   Models: {', '.join(models)}")

    run_model_benchmark(
        project_path=project_path,
        query=args.query,
        models=models,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
