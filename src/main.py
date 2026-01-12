#!/usr/bin/env python3
"""
Recursive Language Model (RLM) - Unified CLI

Usage:
    uv run src/main.py                        # Interactive mode
    uv run src/main.py -q 1 -d 1              # Run query 1 on NSMC
    uv run src/main.py -q 8 -d 3 -s 500k      # Run Law query on Law Insider
    uv run src/main.py --benchmark -d 3 -q 9  # Benchmark comparison
    uv run src/main.py --list                 # List available queries
"""

import argparse
import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402
from termcolor import colored  # noqa: E402

from src.datasets import CONTEXT_SIZES, DATASETS, TEST_QUERIES  # noqa: E402

# Load env variables
load_dotenv(dotenv_path=".env.local")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recursive Language Model (RLM) - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run src/main.py                          # Interactive mode
  uv run src/main.py -q 1                     # Run query 1
  uv run src/main.py -d 3 -q 8 -s 500k        # Law dataset, query 8, 500K context
  uv run src/main.py --benchmark -d 1 -q 1    # Benchmark: Baseline vs Optimized
  uv run src/main.py --benchmark --baseline   # Benchmark: Baseline only
  uv run src/main.py --list                   # List available queries
  uv run src/main.py --list-datasets          # List available datasets
        """,
    )

    # Query options
    parser.add_argument(
        "-q", "--query", type=str, help="Query number (1-10) or custom query string"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        default="1",
        help="Dataset: 1=NSMC, 2=Wiki, 3=Law Insider",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        choices=list(CONTEXT_SIZES.keys()),
        default="100k",
        help="Context size (default: 100k)",
    )

    # Mode options
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode (compare Baseline vs Optimized)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="In benchmark mode: run baseline only",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="In benchmark mode: run optimized only",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Save benchmark results to JSON file"
    )

    # Other options
    parser.add_argument(
        "--list", action="store_true", help="List available test queries and exit"
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="List available datasets and exit"
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable RestrictedPython sandbox (safer but slower)",
    )

    return parser.parse_args()


def list_queries():
    """Display available test queries."""
    print(colored("\n📋 Available Test Queries:", "cyan", attrs=["bold"]))
    print("-" * 70)

    for key, info in TEST_QUERIES.items():
        print(f"  [{key:>2}] {info['name']}")
        print(f"       {info['description']}")
        if info["query"]:
            query_preview = (
                info["query"][:55] + "..." if len(info["query"]) > 55 else info["query"]
            )
            print(colored(f"       → {query_preview}", "dark_grey"))
    print()


def list_datasets():
    """Display available datasets."""
    print(colored("\n📂 Available Datasets:", "cyan", attrs=["bold"]))
    print("-" * 50)

    for key, info in DATASETS.items():
        status = "✅" if os.path.exists(info["path"]) else "❌ (not found)"
        print(f"  [{key}] {info['name']}")
        print(f"      Path: {info['path']} {status}")
    print()


def select_query_interactive() -> str:
    """Display query options and let user select (interactive mode)."""
    print(colored("\n📋 테스트 쿼리 선택:", "cyan", attrs=["bold"]))
    print("-" * 60)

    for key, info in TEST_QUERIES.items():
        print(f"  [{key}] {info['name']}")
        print(f"      └─ {info['description']}")

    print("-" * 60)
    choice = input(colored("선택 (1-10): ", "yellow")).strip()

    if choice not in TEST_QUERIES:
        print(colored("잘못된 선택. 기본값 1 사용.", "red"))
        choice = "1"

    selected = TEST_QUERIES[choice]

    if selected["query"] is None:
        custom = input(colored("질문을 입력하세요: ", "yellow")).strip()
        return custom if custom else TEST_QUERIES["1"]["query"]

    print(colored(f"\n선택된 쿼리: {selected['name']}", "green"))
    return selected["query"]


def select_size_interactive(full_length: int) -> int:
    """Select context size interactively."""
    print(colored("\n📊 컨텍스트 크기 선택:", "cyan"))
    print("  [1] 100K chars (기본, 빠름)")
    print("  [2] 500K chars (중간)")
    print("  [3] 1M chars (대용량)")
    print("  [4] 전체 사용")

    size_choice = input(colored("선택 (1-4, 기본=1): ", "yellow")).strip() or "1"
    limits = {"1": 100000, "2": 500000, "3": 1000000, "4": full_length}
    return limits.get(size_choice, 100000)


def load_context(data_file: str, data_url: str) -> str:
    """Download and load context data."""
    if not os.path.exists(data_file):
        if data_url is None:
            print(colored(f"Error: Local file {data_file} not found.", "red"))
            print(
                colored(
                    "Hint: Run 'uv run src/extract_documents.py' for Law Insider dataset.",
                    "yellow",
                )
            )
            sys.exit(1)

        print(colored(f"Downloading {data_file}...", "yellow"))
        import urllib.request

        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                sys.stdout.write(f"\r{percent:5.1f}% {readsofar:,} / {totalsize:,}")
                if readsofar >= totalsize:
                    sys.stdout.write("\n")

        urllib.request.urlretrieve(data_url, data_file, reporthook)
        print(colored("Download complete.", "green"))

    with open(data_file, "r", encoding="utf-8") as f:
        return f.read()


def run_optimized_mode(context: str, query: str, use_sandbox: bool = False):
    """Run optimized RLM only."""
    from src.rlm_optimized import RLMAgent

    agent = RLMAgent()

    if use_sandbox:
        print(colored("⚠️  Sandbox mode enabled (RestrictedPython)", "yellow"))
        agent.use_sandbox = True

    final_answer = agent.run(context, query)

    print(colored("\n" + "═" * 60, "green"))
    print(colored("📌 Final Answer from RLM:", "green", attrs=["bold"]))
    print(colored("═" * 60, "green"))
    print(final_answer)

    # Print stats
    if hasattr(agent, "recursion_guard"):
        stats = agent.recursion_guard.get_stats()
        print(colored(f"\n📈 Recursion Stats: {stats}", "cyan"))

    print(
        colored(
            f"💰 Estimated Cost: ${agent.stats.get('estimated_cost', 0):.4f}", "cyan"
        )
    )


def run_benchmark_mode(
    context: str,
    query: str,
    run_baseline: bool = True,
    run_optimized: bool = True,
    output_file: str = None,
):
    """Run benchmark comparison."""
    from src.benchmark import run_benchmark

    run_benchmark(
        context=context,
        query=query,
        run_baseline=run_baseline,
        run_optimized=run_optimized,
        output_file=output_file,
    )


def main():
    args = parse_args()

    # List queries and exit
    if args.list:
        list_queries()
        return

    # List datasets and exit
    if args.list_datasets:
        list_datasets()
        return

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(colored("Error: GEMINI_API_KEY not found in .env.local", "red"))
        sys.exit(1)

    # Header
    mode_text = "Benchmark Mode" if args.benchmark else "Optimized Mode"
    print(colored("═" * 60, "green"))
    print(
        colored(
            f"  Recursive Language Model (RLM) - {mode_text}", "green", attrs=["bold"]
        )
    )
    print(colored("═" * 60, "green"))

    # Load dataset
    dataset_info = DATASETS.get(args.dataset, DATASETS["1"])
    data_file = dataset_info["path"]
    data_url = dataset_info["url"]

    print(colored(f"Dataset: {dataset_info['name']}", "cyan"))
    full_text = load_context(data_file, data_url)

    # Determine context size
    if args.query:
        # CLI mode
        context_limit = CONTEXT_SIZES.get(args.size)
        if context_limit is None:
            context_limit = len(full_text)
    else:
        # Interactive mode
        context_limit = select_size_interactive(len(full_text))

    if context_limit > len(full_text):
        context_limit = len(full_text)

    sample_context = full_text[:context_limit]
    print(f"Context loaded: {len(sample_context):,} characters")

    # Determine query
    if args.query:
        # CLI mode
        if args.query in TEST_QUERIES and TEST_QUERIES[args.query]["query"]:
            query = TEST_QUERIES[args.query]["query"]
            print(colored(f"Query: {TEST_QUERIES[args.query]['name']}", "green"))
        else:
            query = args.query  # Custom query string
            print(colored(f"Custom Query: {query[:60]}...", "green"))
    else:
        # Interactive mode
        query = select_query_interactive()

    # Run
    if args.benchmark:
        # Determine what to run
        if args.baseline and not args.optimized:
            run_baseline, run_optimized = True, False
        elif args.optimized and not args.baseline:
            run_baseline, run_optimized = False, True
        else:
            run_baseline, run_optimized = True, True

        run_benchmark_mode(
            context=sample_context,
            query=query,
            run_baseline=run_baseline,
            run_optimized=run_optimized,
            output_file=args.output,
        )
    else:
        run_optimized_mode(sample_context, query, use_sandbox=args.sandbox)


if __name__ == "__main__":
    main()
