#!/usr/bin/env python3
"""
RLM - Recursive Language Model

Main entry point. Launches the TUI interface.

Usage:
    python -m rlm              # Launch TUI
    python -m rlm --help       # Show help
"""

import argparse

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RLM - Recursive Language Model with Dynamic File Access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands in TUI:
  /project        List available projects
  /project <N>    Select project N
  /model          List available models
  /model <name>   Switch to model
  /help           Show help
  /clear          Clear chat

Benchmark (separate CLI):
  python -m rlm.cli.benchmark --help
        """,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.version:
        print("RLM v2.0.0 - Dynamic File Access")
        return

    # Launch TUI
    from rlm.tui.app import main as tui_main

    tui_main()


if __name__ == "__main__":
    main()
