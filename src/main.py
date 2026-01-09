import argparse
import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402
from termcolor import colored  # noqa: E402

from src.rlm import RLMAgent  # noqa: E402

# Load env variables
load_dotenv(dotenv_path=".env.local")


# ============================================================================
# TEST QUERIES - Add new queries here!
# ============================================================================
TEST_QUERIES = {
    "1": {
        "name": "긍정 단어 분석 (Simple)",
        "query": "이 데이터셋에서 가장 많이 등장하는 긍정적인 단어 3개를 찾아줘. 그리고 2023년이라는 숫자가 포함된 리뷰가 있는지 확인해줘.",
        "description": "단순 집계 작업 - llm_query_batch 사용 예상",
    },
    "2": {
        "name": "감정 분포 분석 (Medium)",
        "query": "긍정(label=1)과 부정(label=0) 리뷰의 평균 길이를 비교하고, 각각에서 가장 자주 사용되는 감정 표현 패턴을 분석해줘.",
        "description": "비교 분석 - 약간의 복잡도",
    },
    "3": {
        "name": "섹션별 요약 (Complex - RLM 재귀 권장)",
        "query": "데이터를 1000개씩 5개 섹션으로 나누고, 각 섹션별로 '주요 감정 키워드'와 '대표 리뷰'를 요약해줘. 그리고 전체적인 트렌드를 종합해줘.",
        "description": "복잡한 다단계 작업 - RLM() 재귀 호출 권장",
    },
    "4": {
        "name": "비교 분석 (Complex - RLM 재귀 권장)",
        "query": "긍정 리뷰 500개와 부정 리뷰 500개를 각각 분석해서, 긍정에서만 나타나는 단어와 부정에서만 나타나는 단어를 찾고, 그 차이를 설명해줘.",
        "description": "비교 대조 분석 - RLM() 재귀 호출 권장",
    },
    "5": {
        "name": "Custom Query",
        "query": None,
        "description": "직접 질문 입력",
    },
}

CONTEXT_SIZES = {
    "100k": 100000,
    "500k": 500000,
    "1m": 1000000,
    "full": None,  # Will be set to full length
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recursive Language Model (RLM) Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run src/main.py                        # Interactive mode
  uv run src/main.py -q 1                   # Run query 1 (Simple)
  uv run src/main.py -q 3 -s 500k           # Run query 3 with 500K context
  uv run src/main.py --query "질문" -s 1m   # Custom query with 1M context
  uv run src/main.py --list                 # List available queries
        """,
    )
    parser.add_argument(
        "-q", "--query", type=str, help="Query number (1-4) or custom query string"
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        choices=["100k", "500k", "1m", "full"],
        default="100k",
        help="Context size (default: 100k)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available test queries and exit"
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable RestrictedPython sandbox (safer but slower)",
    )
    return parser.parse_args()


def select_query_interactive() -> str:
    """Display query options and let user select (interactive mode)."""
    print(colored("\n📋 테스트 쿼리 선택:", "cyan", attrs=["bold"]))
    print("-" * 60)

    for key, info in TEST_QUERIES.items():
        print(f"  [{key}] {info['name']}")
        print(f"      └─ {info['description']}")

    print("-" * 60)
    choice = input(colored("선택 (1-5): ", "yellow")).strip()

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
    print("  [4] 전체 사용 (~14MB)")

    size_choice = input(colored("선택 (1-4, 기본=1): ", "yellow")).strip() or "1"
    limits = {"1": 100000, "2": 500000, "3": 1000000, "4": full_length}
    return limits.get(size_choice, 100000)


def load_context(data_file: str, data_url: str) -> str:
    """Download and load context data."""
    if not os.path.exists(data_file):
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


def main():
    args = parse_args()

    # List queries and exit
    if args.list:
        print(colored("\n📋 Available Test Queries:", "cyan", attrs=["bold"]))
        for key, info in TEST_QUERIES.items():
            print(f"  [{key}] {info['name']}")
            print(f"      {info['description']}")
            if info["query"]:
                print(f"      Query: {info['query'][:60]}...")
        return

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(colored("Error: GEMINI_API_KEY not found in .env.local", "red"))
        sys.exit(1)

    print(colored("═" * 60, "green"))
    print(colored("  Recursive Language Model (RLM) Runner", "green", attrs=["bold"]))
    print(colored("═" * 60, "green"))

    # Load context
    data_file = "ratings_train.txt"
    data_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    full_text = load_context(data_file, data_url)

    # Determine context size
    if args.query:
        # CLI mode
        context_limit = CONTEXT_SIZES.get(args.size, 100000)
        if context_limit is None:
            context_limit = len(full_text)
    else:
        # Interactive mode
        context_limit = select_size_interactive(len(full_text))

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

    # Run RLM
    agent = RLMAgent()

    # Enable sandbox if requested (Phase 3 feature)
    if args.sandbox:
        print(colored("⚠️  Sandbox mode enabled (RestrictedPython)", "yellow"))
        agent.use_sandbox = True

    final_answer = agent.run(sample_context, query)

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


if __name__ == "__main__":
    main()
