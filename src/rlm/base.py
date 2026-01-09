"""
Base RLM Agent - Abstract interface for RLM implementations.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.llm_client import GeminiClient
from src.logger_config import setup_logger
from src.rlm.config import PRICING


@dataclass
class RLMStats:
    """Statistics for a single RLM run."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    estimated_cost: float = 0.0
    llm_calls: int = 0
    steps: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class BaseRLMAgent(ABC):
    """
    Abstract base class for RLM agents.

    Both baseline and optimized agents inherit from this class,
    ensuring consistent interfaces for benchmarking.
    """

    VERSION: str = "base"
    PRICING = PRICING

    def __init__(self, output_dir: str = "."):
        self.client = GeminiClient()
        self.max_steps = 10
        self.chat = None
        self.stats = RLMStats()
        self.logger, self.log_file = setup_logger()

    @abstractmethod
    def run(self, context_text: str, user_query: str) -> str:
        """Execute the RLM agent on the given context and query."""
        pass

    @abstractmethod
    def _create_repl(self, context_text: str) -> Any:
        """Create the appropriate REPL for this agent type."""
        pass

    def _update_stats(self, input_tokens: int, output_tokens: int) -> None:
        """Update token and cost statistics."""
        if not input_tokens or not output_tokens:
            return
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens
        self.stats.total_tokens += input_tokens + output_tokens
        self.stats.llm_calls += 1

        # Calculate cost
        model = self.client.model_name
        pricing = self.PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens / 1000) * pricing["input"] + (
            output_tokens / 1000
        ) * pricing["output"]
        self.stats.estimated_cost += cost

    def _start_run(self, context_text: str, user_query: str) -> None:
        """Initialize run statistics."""
        self.stats = RLMStats()
        self.stats.start_time = time.time()
        self.logger.info(f"--- Starting {self.VERSION} RLM ---")
        self.logger.info(f"Query: {user_query}")
        self.logger.info(f"Context Length: {len(context_text)} chars")

    def _finish_run(self) -> None:
        """Finalize run and log statistics."""
        self.stats.end_time = time.time()

        self.logger.info("\n" + "=" * 50)
        self.logger.info(f"{self.VERSION} RLM Execution Complete")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Duration: {self.stats.duration:.2f}s")
        self.logger.info(
            f"Total Tokens: {self.stats.total_tokens:,} "
            f"(In: {self.stats.total_input_tokens:,}, Out: {self.stats.total_output_tokens:,})"
        )
        self.logger.info(f"LLM Calls: {self.stats.llm_calls}")
        self.logger.info(f"Estimated Cost: ${self.stats.estimated_cost:.4f}")
        self.logger.info(f"Full log saved to: {self.log_file}")

    def _extract_code_blocks(self, text: str) -> list[str]:
        """
        Extract Python code blocks from markdown text.
        Handles cases where ``` appears inside string literals in the code.
        """
        blocks = []
        lines = text.split("\n")
        in_code_block = False
        current_block = []

        for line in lines:
            stripped = line.strip()

            if not in_code_block:
                if stripped.startswith("```python") or stripped == "```python":
                    in_code_block = True
                    current_block = []
            else:
                if stripped == "```":
                    if current_block:
                        blocks.append("\n".join(current_block))
                    in_code_block = False
                    current_block = []
                else:
                    current_block.append(line)

        if in_code_block and current_block:
            blocks.append("\n".join(current_block))

        return [b.strip() for b in blocks if b.strip()]
