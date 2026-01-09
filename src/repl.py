import contextlib
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List


class PythonREPL:
    def __init__(self, context_str: str, llm_query_func: Callable[[str], str]):
        self.context_str = context_str
        self.llm_query_func = llm_query_func
        self._cache: Dict[str, str] = {}  # Cache for query results

        self.scope: Dict[str, Any] = {
            "context": context_str,
            "llm_query": self.llm_query_wrapper,
            "llm_query_batch": self.llm_query_batch,  # New: parallel batch calls
            "estimate_chunk_size": self.estimate_chunk_size,  # New: dynamic chunking
            "print": print,
        }

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a cache key from prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def llm_query_wrapper(self, prompt: str) -> str:
        """Wrapper to be called from within the REPL. Includes caching."""
        cache_key = self._get_cache_key(prompt)

        # Check cache first
        if cache_key in self._cache:
            print(f"\n[REPL] Cache HIT for prompt: {prompt[:50]}...")
            return self._cache[cache_key]

        print(f"\n[REPL] Calling sub-LLM with prompt: {prompt[:50]}...")
        result = self.llm_query_func(prompt)

        # Store in cache
        self._cache[cache_key] = result
        return result

    def llm_query_batch(self, prompts: List[str], max_workers: int = 4) -> List[str]:
        """
        Execute multiple LLM queries in parallel.
        Returns results in the same order as input prompts.
        """
        print(
            f"\n[REPL] Batch processing {len(prompts)} prompts with {max_workers} workers..."
        )

        results = [None] * len(prompts)

        def process_single(idx: int, prompt: str) -> tuple:
            cache_key = self._get_cache_key(prompt)

            # Check cache
            if cache_key in self._cache:
                print(f"  [Batch {idx+1}] Cache HIT")
                return (idx, self._cache[cache_key])

            print(f"  [Batch {idx+1}] Calling LLM...")
            result = self.llm_query_func(prompt)
            self._cache[cache_key] = result
            return (idx, result)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single, i, p): i for i, p in enumerate(prompts)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        print(f"[REPL] Batch complete. {len(results)} results collected.")
        return results

    def estimate_chunk_size(
        self,
        items: List[Any],
        target_chunks: int = 4,
        min_chunk_size: int = 10,
        max_chunk_size: int = 500,
    ) -> int:
        """
        Estimate optimal chunk size for a list of items.

        Args:
            items: List of items to chunk
            target_chunks: Desired number of chunks (default: 4 for parallel processing)
            min_chunk_size: Minimum items per chunk
            max_chunk_size: Maximum items per chunk

        Returns:
            Recommended chunk size
        """
        if not items:
            return min_chunk_size

        total = len(items)
        ideal_size = max(1, total // target_chunks)

        # Clamp to min/max bounds
        chunk_size = max(min_chunk_size, min(ideal_size, max_chunk_size))

        actual_chunks = (total + chunk_size - 1) // chunk_size
        print(
            f"[REPL] Estimated chunk size: {chunk_size} (will create ~{actual_chunks} chunks for {total} items)"
        )

        return chunk_size

    def execute(self, code: str) -> str:
        """Executes the given python code and returns the stdout."""
        stdout_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, self.scope)

            output = stdout_buffer.getvalue()
            return output if output else "(No output)"

        except Exception as e:
            return f"Runtime Error: {e}"
