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
        project_path: str = None,
        context_str: str = None,  # For recursive calls with extracted context
        llm_query_func: Callable[[str], str] = None,
        rlm_recursive_func: Optional[Callable[[str, str, int], str]] = None,
        recursion_guard: Optional[RecursionGuard] = None,
        current_depth: int = 0,
        use_sandbox: bool = False,
        sub_step_callback: Optional[Callable[[str, str], None]] = None,
    ):
        from pathlib import Path

        # Dual mode: project_path for file exploration, context_str for recursive calls
        self.project_path = Path(project_path) if project_path else None
        self.context_str = context_str

        self.llm_query_func = llm_query_func
        self.rlm_recursive_func = rlm_recursive_func
        self.recursion_guard = recursion_guard or RecursionGuard()
        self.current_depth = current_depth
        self.use_sandbox = use_sandbox and SANDBOX_AVAILABLE
        self._cache: Dict[str, str] = {}  # LLM cache
        self._file_cache: Dict[str, str] = {}  # Parsed file content cache
        self.sub_step_callback = sub_step_callback

        # Build scope based on mode
        self.scope: Dict[str, Any] = {
            # LLM tools (always available)
            "llm_query": self.llm_query_wrapper,
            "llm_query_batch": self.llm_query_batch,
            "RLM": self.rlm_recursive_wrapper,
            "estimate_chunk_size": self.estimate_chunk_size,
            "print": print,
            **self.SAFE_MODULES,
        }

        if self.project_path:
            # File exploration mode
            self.scope.update(
                {
                    "project_path": str(self.project_path),
                    "list_files": self.list_files,
                    "read_file": self.read_file,
                    "search_files": self.search_files,
                    "get_file_info": self.get_file_info,
                }
            )

        if self.context_str:
            # Context mode (for recursive calls)
            self.scope["context"] = self.context_str

    # =========================================================================
    # File System Tools for Dynamic Exploration
    # =========================================================================

    def list_files(self) -> List[str]:
        """List all supported files in the project folder."""
        from rlm.parsers.loader import PARSERS, TEXT_EXTENSIONS

        self._report_sub("list_files", "Scanning project...", "", False)

        supported = set(PARSERS.keys()) | TEXT_EXTENSIONS
        files = []

        for f in sorted(self.project_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in supported:
                if not f.name.startswith(".") and not f.name.startswith("~"):
                    rel_path = f.relative_to(self.project_path)
                    files.append(str(rel_path))

        self._report_sub("list_files", f"Found {len(files)} files", "", True)
        return files

    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """Get metadata about a file."""
        file_path = self.project_path / filename

        if not file_path.exists():
            return {"error": f"File not found: {filename}"}

        if not file_path.is_relative_to(self.project_path):
            return {"error": "Access denied: outside project folder"}

        stat = file_path.stat()

        # Count lines for text files
        line_count = None
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md", ".csv", ".json", ".xml", ".html"}:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                pass

        return {
            "name": filename,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 1),
            "format": suffix,
            "line_count": line_count,
        }

    def read_file(
        self, filename: str, start_line: int = 0, max_lines: int = 100
    ) -> str:
        """
        Read file content with line range support.

        Args:
            filename: Relative path to file
            start_line: Starting line (0-indexed)
            max_lines: Maximum lines to read (default 100, max 200)

        Returns:
            File content as string
        """
        self._report_sub("read_file", f"Reading {filename}...", "", False)
        from rlm.parsers.loader import PARSERS, TEXT_EXTENSIONS

        # Limit max_lines to prevent context explosion
        max_lines = min(max_lines, 200)

        file_path = self.project_path / filename

        if not file_path.exists():
            return f"[Error: File not found: {filename}]"

        if not file_path.is_relative_to(self.project_path):
            return "[Error: Access denied: outside project folder]"

        suffix = file_path.suffix.lower()

        try:
            # PDF, DOCX, PPTX - use parsers
            if suffix in PARSERS:
                content = PARSERS[suffix](file_path)
                lines = content.split("\n")
                selected = lines[start_line : start_line + max_lines]
                return "\n".join(selected)

            # Text files - read directly with encoding fallback
            elif suffix in TEXT_EXTENSIONS:
                for encoding in ["utf-8", "cp949", "euc-kr", "latin-1"]:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            lines = []
                            for i, line in enumerate(f):
                                if i < start_line:
                                    continue
                                if i >= start_line + max_lines:
                                    break
                                lines.append(line.rstrip("\n"))
                        self._report_sub("read_file", f"{len(lines)} lines", "", True)
                        return "\n".join(lines)
                    except UnicodeDecodeError:
                        continue
                return f"[Error: Failed to decode {filename}]"

            else:
                return f"[Error: Unsupported format: {suffix}]"

        except Exception as e:
            return f"[Error reading {filename}: {e}]"

    def search_files(self, keyword: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for keyword across all files in project.

        Args:
            keyword: Search term (case-insensitive)
            max_results: Maximum number of results

        Returns:
            List of matches with file, line_number, and snippet
        """
        from rlm.parsers.loader import PARSERS, TEXT_EXTENSIONS

        self._report_sub("search", f"Searching '{keyword}'...", "", False)

        results = []
        keyword_lower = keyword.lower()

        for filename in self.list_files():
            file_path = self.project_path / filename
            suffix = file_path.suffix.lower()

            try:
                # Check cache first
                if filename in self._file_cache:
                    content = self._file_cache[filename]
                else:
                    content = None

                    # Parse binary formats (PDF, DOCX, PPTX)
                    if suffix in PARSERS:
                        try:
                            content = PARSERS[suffix](file_path)
                        except Exception:
                            continue

                    # Text files
                    elif suffix in TEXT_EXTENSIONS:
                        for encoding in ["utf-8", "cp949", "latin-1"]:
                            try:
                                content = file_path.read_text(encoding=encoding)
                                break
                            except UnicodeDecodeError:
                                continue

                    if content:
                        # Cache parsed content
                        self._file_cache[filename] = content

                if not content:
                    continue

                # Search in content
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    if keyword_lower in line.lower():
                        results.append(
                            {
                                "file": filename,
                                "line": line_num,
                                "snippet": line.strip()[:200],
                            }
                        )
                        if len(results) >= max_results:
                            return results

            except Exception:
                continue

        self._report_sub("search", f"Found {len(results)} results", "", True)
        return results

    def _report_sub(
        self, sub_id: str, title: str, content: str = "", is_complete: bool = False
    ):
        """Report sub-step progress if callback is available."""
        if self.sub_step_callback:
            try:
                self.sub_step_callback(sub_id, title, content, is_complete)
            except Exception:
                pass
        print(f"[REPL] {sub_id}: {title[:60]}...")

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a cache key from prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def llm_query_wrapper(self, prompt: str) -> str:
        """Wrapper for simple sub-LLM calls. Includes caching."""
        # Limit prompt length to prevent huge API calls (max ~10K chars)
        max_prompt_len = 10000
        if len(prompt) > max_prompt_len:
            prompt = (
                prompt[:max_prompt_len] + f"\n... [Truncated from {len(prompt)} chars]"
            )

        cache_key = self._get_cache_key(prompt)
        prompt_preview = prompt[:60].replace("\n", " ")

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self._report_sub("Sub-LLM", "Cache HIT", cached[:80], True)
            return cached

        self._report_sub("Sub-LLM", f"Calling: {prompt_preview}", "", False)
        result = self.llm_query_func(prompt)
        self._report_sub("Sub-LLM", "Complete", result[:100].replace("\n", " "), True)

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
        self._report_sub("RLM-Recursive", f"Starting: {query[:50]}", "", False)
        result = self.rlm_recursive_func(sub_context, query, next_depth)

        self._report_sub(
            "RLM-Recursive", "Complete", result[:80].replace("\n", " "), True
        )
        return result

    def llm_query_batch(self, prompts: List[str], max_workers: int = 4) -> List[str]:
        """Execute multiple LLM queries in parallel."""
        self._report_sub("Batch", f"Processing {len(prompts)} prompts", "", False)

        results = [None] * len(prompts)
        completed_count = [0]  # Use list for mutable counter in closure

        def process_single(idx: int, prompt: str) -> tuple:
            sub_id = f"Batch[{idx+1}]"
            cache_key = self._get_cache_key(prompt)
            prompt_preview = prompt[:50].replace("\n", " ")

            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                completed_count[0] += 1
                self._report_sub(sub_id, "Cache HIT", cached_result[:100], True)
                return (idx, cached_result)

            self._report_sub(sub_id, f"Calling: {prompt_preview}", "", False)
            result = self.llm_query_func(prompt)
            self._cache[cache_key] = result
            completed_count[0] += 1
            self._report_sub(
                sub_id,
                f"✓ ({completed_count[0]}/{len(prompts)})",
                result[:100].replace("\n", " "),
                True,
            )
            return (idx, result)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single, i, p): i for i, p in enumerate(prompts)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        self._report_sub("Batch", f"Complete: {len(results)} results", "", True)
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
                exec(byte_code, restricted_globals)

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
