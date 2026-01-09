import sys
import io
import contextlib
from typing import Any, Dict, Callable

class PythonREPL:
    def __init__(self, context_str: str, llm_query_func: Callable[[str], str]):
        self.context_str = context_str
        self.llm_query_func = llm_query_func
        self.scope: Dict[str, Any] = {
            "context": context_str,
            "llm_query": self.llm_query_wrapper,
            "print": print  # Explicitly ensure print is available
        }
    
    def llm_query_wrapper(self, prompt: str) -> str:
        """Wrapper to be called from within the REPL."""
        print(f"\n[REPL] Calling sub-LLM with prompt: {prompt[:50]}...")
        return self.llm_query_func(prompt)

    def execute(self, code: str) -> str:
        """Executes the given python code and returns the stdout."""
        # Capture stdout
        stdout_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                # Execute code in the defined scope
                exec(code, self.scope)
            
            output = stdout_buffer.getvalue()
            return output if output else "(No output)"
            
        except Exception as e:
            return f"Runtime Error: {e}"
