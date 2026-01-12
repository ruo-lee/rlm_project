"""
Baseline REPL - Original paper implementation without optimizations.

No caching, no parallel batch processing, no dynamic chunking.
Only provides llm_query and RLM for recursive calls.
"""

import contextlib
import io
from typing import Any, Callable, Dict, Optional


class BaselineREPL:
    """
    Simple Python REPL for baseline RLM.
    Matches the original paper's REPL capabilities.
    """

    def __init__(
        self,
        context_str: str,
        llm_query_func: Callable[[str], str],
        rlm_recursive_func: Optional[Callable[[str, str, int], str]] = None,
        current_depth: int = 0,
    ):
        self.context_str = context_str
        self.llm_query_func = llm_query_func
        self.rlm_recursive_func = rlm_recursive_func
        self.current_depth = current_depth

        # Minimal scope - matches original paper
        self.scope: Dict[str, Any] = {
            "context": context_str,
            "llm_query": self._llm_query_wrapper,
            "RLM": self._rlm_wrapper,
            "print": print,
        }

    def _llm_query_wrapper(self, prompt: str) -> str:
        """Simple wrapper - NO CACHING."""
        print(f"\n[REPL] Calling sub-LLM with prompt: {prompt[:50]}...")
        return self.llm_query_func(prompt)

    def _rlm_wrapper(self, sub_context: str, query: str) -> str:
        """Recursive RLM call."""
        if self.rlm_recursive_func is None:
            print("[REPL] WARNING: RLM not available, falling back to llm_query")
            return self._llm_query_wrapper(
                f"Context:\n{sub_context[:5000]}\n\nQuery: {query}"
            )

        next_depth = self.current_depth + 1
        print(f"\n[REPL] *** RECURSIVE RLM CALL (depth {next_depth}) ***")
        print(f"[REPL] Sub-context: {len(sub_context)} chars, Query: {query[:50]}...")

        result = self.rlm_recursive_func(sub_context, query, next_depth)

        print(f"[REPL] *** RECURSIVE CALL COMPLETE (depth {next_depth}) ***")
        return result

    def execute(self, code: str) -> str:
        """Execute Python code and return stdout."""
        stdout_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, self.scope)

            output = stdout_buffer.getvalue()
            return output if output else "(No output)"

        except Exception as e:
            return f"Runtime Error: {e}"
