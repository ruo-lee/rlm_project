import contextlib
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

# RestrictedPython for secure code execution
try:
    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.Eval import default_guarded_getitem
    from RestrictedPython.Guards import guarded_iter_unpack_sequence

    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    SANDBOX_AVAILABLE = False


class RecursionGuard:
    """Prevents infinite recursion and circular references."""

    def __init__(self, max_depth: int = 5, max_total_calls: int = 50):
        self.max_depth = max_depth
        self.max_total_calls = max_total_calls
        self.call_count = 0
        self.call_hashes: Dict[int, set] = {}  # depth -> set of task hashes

    def check(self, depth: int, task_hash: str) -> None:
        """Check if recursion is safe. Raises RecursionError if not."""
        self.call_count += 1

        # Check max depth
        if depth > self.max_depth:
            raise RecursionError(f"Max recursion depth ({self.max_depth}) exceeded")

        # Check total calls
        if self.call_count > self.max_total_calls:
            raise RecursionError(f"Max total calls ({self.max_total_calls}) exceeded")

        # Check circular reference (same task at same depth)
        if depth not in self.call_hashes:
            self.call_hashes[depth] = set()

        if task_hash in self.call_hashes[depth]:
            raise RecursionError(f"Circular recursion detected at depth {depth}")

        self.call_hashes[depth].add(task_hash)

    def get_stats(self) -> dict:
        """Return recursion statistics."""
        return {
            "total_calls": self.call_count,
            "max_depth_reached": (
                max(self.call_hashes.keys()) if self.call_hashes else 0
            ),
        }


class PythonREPL:
    # Safe modules that can be imported in sandbox mode
    SAFE_MODULES = {
        "json": __import__("json"),
        "re": __import__("re"),
        "math": __import__("math"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
    }

    def __init__(
        self,
        context_str: str,
        llm_query_func: Callable[[str], str],
        rlm_recursive_func: Optional[Callable[[str, str, int], str]] = None,
        recursion_guard: Optional[RecursionGuard] = None,
        current_depth: int = 0,
        use_sandbox: bool = False,
    ):
        self.context_str = context_str
        self.llm_query_func = llm_query_func
        self.rlm_recursive_func = rlm_recursive_func
        self.recursion_guard = recursion_guard or RecursionGuard()
        self.current_depth = current_depth
        self.use_sandbox = use_sandbox and SANDBOX_AVAILABLE
        self._cache: Dict[str, str] = {}

        self.scope: Dict[str, Any] = {
            "context": context_str,
            "llm_query": self.llm_query_wrapper,
            "llm_query_batch": self.llm_query_batch,
            "RLM": self.rlm_recursive_wrapper,
            "estimate_chunk_size": self.estimate_chunk_size,
            "print": print,
            # Add safe modules to scope
            **self.SAFE_MODULES,
        }

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a cache key from prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def llm_query_wrapper(self, prompt: str) -> str:
        """Wrapper for simple sub-LLM calls. Includes caching."""
        cache_key = self._get_cache_key(prompt)

        if cache_key in self._cache:
            print(f"\n[REPL] Cache HIT for prompt: {prompt[:50]}...")
            return self._cache[cache_key]

        print(f"\n[REPL] Calling sub-LLM with prompt: {prompt[:50]}...")
        result = self.llm_query_func(prompt)

        self._cache[cache_key] = result
        return result

    def rlm_recursive_wrapper(self, sub_context: str, query: str) -> str:
        """
        TRUE RECURSIVE RLM CALL!
        This executes a full RLM loop on a sub-problem.

        Args:
            sub_context: The context/data for the sub-problem
            query: The question to answer about sub_context

        Returns:
            The final answer from the recursive RLM call
        """
        if self.rlm_recursive_func is None:
            print(
                "[REPL] WARNING: RLM recursive function not available. Falling back to llm_query."
            )
            return self.llm_query_wrapper(
                f"Context:\n{sub_context[:5000]}\n\nQuery: {query}"
            )

        # Check recursion safety
        task_hash = self._get_cache_key(f"{sub_context[:1000]}:{query}")
        next_depth = self.current_depth + 1

        try:
            self.recursion_guard.check(next_depth, task_hash)
        except RecursionError as e:
            print(f"[REPL] Recursion limit: {e}. Falling back to llm_query.")
            return self.llm_query_wrapper(
                f"Context:\n{sub_context[:5000]}\n\nQuery: {query}"
            )

        print(f"\n[REPL] *** RECURSIVE RLM CALL (depth {next_depth}) ***")
        print(f"[REPL] Sub-context length: {len(sub_context)}, Query: {query[:50]}...")

        # Execute full RLM loop recursively
        result = self.rlm_recursive_func(sub_context, query, next_depth)

        print(f"[REPL] *** RECURSIVE CALL COMPLETE (depth {next_depth}) ***")
        return result

    def llm_query_batch(self, prompts: List[str], max_workers: int = 4) -> List[str]:
        """Execute multiple LLM queries in parallel."""
        print(
            f"\n[REPL] Batch processing {len(prompts)} prompts with {max_workers} workers..."
        )

        results = [None] * len(prompts)

        def process_single(idx: int, prompt: str) -> tuple:
            cache_key = self._get_cache_key(prompt)

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
        max_chunk_size: int = 100,
    ) -> int:
        """Estimate optimal chunk size for a list of items."""
        if not items:
            return min_chunk_size

        total = len(items)
        ideal_size = max(1, total // target_chunks)
        chunk_size = max(min_chunk_size, min(ideal_size, max_chunk_size))

        actual_chunks = (total + chunk_size - 1) // chunk_size
        print(
            f"[REPL] Estimated chunk size: {chunk_size} (~{actual_chunks} chunks for {total} items)"
        )

        return chunk_size

    def execute(self, code: str) -> str:
        """Executes the given python code and returns the stdout."""
        if self.use_sandbox:
            return self._execute_sandboxed(code)
        return self._execute_normal(code)

    def _execute_normal(self, code: str) -> str:
        """Standard execution (no sandbox)."""
        stdout_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, self.scope)

            output = stdout_buffer.getvalue()
            if not output:
                return "(No output)"

            # Truncate output to prevent token explosion
            max_len = 2000
            if len(output) > max_len:
                return (
                    output[:max_len]
                    + f"\n... [Output truncated. Total length: {len(output)} chars]"
                )
            return output

        except Exception as e:
            return f"Runtime Error: {e}"

    def _execute_sandboxed(self, code: str) -> str:
        """Execute code in RestrictedPython sandbox."""
        if not SANDBOX_AVAILABLE:
            print("[REPL] WARNING: RestrictedPython not available, using normal exec")
            return self._execute_normal(code)

        stdout_buffer = io.StringIO()

        try:
            # Compile with restrictions
            byte_code = compile_restricted(code, "<user_code>", "exec")

            if byte_code.errors:
                return f"Compilation Error: {byte_code.errors}"

            # Create restricted globals
            restricted_globals = {
                "__builtins__": safe_builtins,
                "_getitem_": default_guarded_getitem,
                "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                "_getiter_": iter,
                "_print_": print,
                # Add our scope items
                **self.scope,
            }

            with contextlib.redirect_stdout(stdout_buffer):
                exec(byte_code.code, restricted_globals)

            output = stdout_buffer.getvalue()
            if not output:
                return "(No output)"

            # Truncate output to prevent token explosion
            max_len = 2000
            if len(output) > max_len:
                return (
                    output[:max_len]
                    + f"\n... [Output truncated. Total length: {len(output)} chars]"
                )
            return output

        except Exception as e:
            return f"Sandbox Error: {e}"
