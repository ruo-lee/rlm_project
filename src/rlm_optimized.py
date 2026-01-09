import time

from google.genai import types
from termcolor import colored

from src.llm_client import GeminiClient
from src.logger_config import setup_logger
from src.repl_optimized import PythonREPL, RecursionGuard

SYSTEM_PROMPT_TEMPLATE = """
You are a Recursive Language Model (RLM).
You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a Python REPL environment.

The REPL environment is initialized with:
1. `context`: A string variable containing the text you need to process.
2. `llm_query(prompt)`: Query a sub-LLM with a single prompt. Results are cached.
3. `llm_query_batch(prompts, max_workers=4)`: Query sub-LLM with multiple prompts IN PARALLEL.
4. `RLM(sub_context, query)`: **TRUE RECURSIVE CALL** - Spawns a NEW complete RLM session!
   - Use this for complex sub-problems that need their own exploration/planning loop.
   - Example: `answer = RLM(document_chunk, "Summarize this section")`
   - The sub-RLM has its own context, can write code, and returns a final answer.
5. `estimate_chunk_size(items, target_chunks=4)`: Helps determine optimal chunk size.
6. `print()`: Use this to see the output of your code.

WHEN TO USE EACH TOOL:
- `llm_query`: Simple questions, extraction, classification (single LLM call)
- `llm_query_batch`: Same simple task on multiple chunks (parallel)
- `RLM()`: Complex sub-problems needing multi-step reasoning (recursive)

Process:
1. EXPLORE: Check the length of `context`, peek at the beginning/end, or search for keywords.
2. PLAN: Decide how to break down the problem.
3. EXECUTE: For simple tasks use llm_query_batch, for complex sub-tasks use RLM().
4. SYNTHESIZE: Gather results from your sub-calls and print them.
5. ANSWER: When you have the answer, print "FINAL ANSWER: [your answer]" to finish.

EFFICIENCY RULES:
- FILTER FIRST: Filter to relevant subset BEFORE heavy processing.
- COMPLETE QUICKLY: Finish within 3-5 steps.
- USE RLM() SPARINGLY: Only for genuinely complex sub-problems.

CRITICAL INSTRUCTIONS:
- ALWAYS wrap your Python code in ```python ... ``` blocks.
- To finish, you MUST print a line starting with "FINAL ANSWER:".

CODE SAFETY RULES:
- NEVER use triple backticks (```) inside your Python code strings.
- For JSON parsing: try `json.loads(response.strip())`, or find first `{` and last `}`.
"""


class RLMAgent:
    # Pricing per 1K tokens (official pricing as of Jan 2026)
    # Source: https://ai.google.dev/pricing
    PRICING = {
        # Gemini 3 Pro Preview: $2/1M input, $12/1M output
        "gemini-3-pro-preview": {"input": 0.002, "output": 0.012},
        # Gemini 3 Flash Preview: $0.50/1M input, $3/1M output
        "gemini-3-flash-preview": {"input": 0.0005, "output": 0.003},
        # Gemini 2.5 Flash: $0.30/1M input, $2.50/1M output
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
        # Gemini 2.0 Flash: $0.10/1M input, $0.40/1M output
        "gemini-2.0-flash-exp": {"input": 0.0001, "output": 0.0004},
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        # Gemini 1.5 models
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    }

    def __init__(self, output_dir: str = "."):
        self.client = GeminiClient()
        self.max_steps = 10
        self.chat_history = []
        self.chat = None

        # Recursion control (shared across recursive calls)
        self.recursion_guard = RecursionGuard(max_depth=5, max_total_calls=50)

        # Execution tree for visualization
        self.execution_tree = []  # List of execution nodes

        # Metrics
        self.stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "start_time": 0,
            "end_time": 0,
            "steps": [],
            "estimated_cost": 0.0,
        }

        # Sandbox mode (disabled by default)
        self.use_sandbox = False

        # Setup Logger
        self.logger, self.log_file = setup_logger()

    def run(self, context_text: str, user_query: str):
        self.stats["start_time"] = time.time()
        self.logger.info("--- Starting RLM ---")
        self.logger.info(f"Query: {user_query}")
        self.logger.info(f"Context Length: {len(context_text)} chars")

        # 1. Initialize REPL with recursive RLM support
        repl = PythonREPL(
            context_str=context_text,
            llm_query_func=self._sub_llm_call,
            rlm_recursive_func=self._recursive_run,
            recursion_guard=self.recursion_guard,
            current_depth=0,
            use_sandbox=self.use_sandbox,
        )

        # 2. Initialize Chat Session
        self.chat = self.client.client.chats.create(
            model=self.client.model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_TEMPLATE
            ),
        )

        user_message = f"Query: {user_query}\n\nContext length: {len(context_text)} chars.\nPlease start by exploring the context."
        next_prompt = user_message

        for step in range(self.max_steps):
            step_start_time = time.time()
            self.logger.info(f"\n=== Step {step + 1}/{self.max_steps} ===")

            # Send message to chat
            try:
                self.logger.info("[Action] Sending prompt to Root LLM...")
                print(colored("[RLM Thought] (Thinking...):", "blue"))

                response_stream = self.chat.send_message_stream(next_prompt)

                response_text = ""
                for chunk in response_stream:
                    if chunk.text:
                        text_chunk = chunk.text
                        response_text += text_chunk
                        print(colored(text_chunk, "blue"), end="", flush=True)

                    # Track usage if available in chunks (usually last chunk)
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        u = chunk.usage_metadata
                        self._update_stats(
                            u.prompt_token_count, u.candidates_token_count
                        )

                print()  # Newline after thought stream

                # Log the full thought after streaming
                self.logger.info(f"[RLM Thought]:\n{response_text}")

            except Exception as e:
                self.logger.error(f"Error during LLM call: {e}")
                return "Error during execution."

            # Parse code
            code_blocks = self._extract_code_blocks(response_text)

            step_info = {
                "step": step + 1,
                "thought": response_text,
                "code_executed": None,
                "output": None,
                "duration": time.time() - step_start_time,
            }

            if not code_blocks:
                if "FINAL ANSWER:" in response_text:
                    self._finish_run()
                    return response_text.split("FINAL ANSWER:")[-1].strip()

                self.logger.warning("No code block found. Asking model to retry.")
                next_prompt = "You didn't provide any code. Please write Python code to inspect the context or output 'FINAL ANSWER:'."
                self.stats["steps"].append(step_info)
                continue

            # Execute code
            full_code = "\n".join(code_blocks)
            self.logger.info(f"[Executing Code]:\n{full_code}")
            step_info["code_executed"] = full_code

            execution_output = repl.execute(full_code)

            # Truncate output for log readability, but full could be in file if separate handler used
            msg_output = (
                execution_output[:2000] + "...(truncated)"
                if len(execution_output) > 2000
                else execution_output
            )
            self.logger.info(f"[Execution Output]:\n{msg_output}")
            step_info["output"] = execution_output

            # Pass output back to LLM
            next_prompt = f"Code Output:\n{execution_output}\n\nBased on this, what is the next step?"

            if "FINAL ANSWER:" in response_text:
                self._finish_run()
                return response_text.split("FINAL ANSWER:")[-1].strip()

            self.stats["steps"].append(step_info)

        self._finish_run()
        return "Max steps reached without final answer."

    def _sub_llm_call(self, prompt: str) -> str:
        """Function exposed to REPL as `llm_query`."""
        self.logger.info(f"[Sub-LLM Call] Prompt: {prompt[:100]}...")
        result = self.client.generate_content(prompt)

        # Track usage from sub-calls
        usage = result.get("usage", {})
        self._update_stats(
            usage.get("prompt_token_count", 0), usage.get("candidates_token_count", 0)
        )

        return result["text"]

    def _recursive_run(self, sub_context: str, query: str, depth: int) -> str:
        """
        TRUE RECURSIVE RLM CALL!
        Executes a complete RLM loop for a sub-problem.

        This is the key innovation from the paper: the model can call itself
        recursively to handle arbitrarily complex sub-tasks.
        """
        self.logger.info(f"\n{'='*20} RECURSIVE RLM (depth {depth}) {'='*20}")
        self.logger.info(f"Sub-query: {query}")
        self.logger.info(f"Sub-context length: {len(sub_context)} chars")

        # Create a new REPL for this recursive call
        repl = PythonREPL(
            context_str=sub_context,
            llm_query_func=self._sub_llm_call,
            rlm_recursive_func=self._recursive_run,  # Allow further recursion
            recursion_guard=self.recursion_guard,  # Shared guard
            current_depth=depth,
        )

        # Create a NEW chat session for this sub-problem
        sub_chat = self.client.client.chats.create(
            model=self.client.model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_TEMPLATE
            ),
        )

        user_message = f"Query: {query}\n\nContext length: {len(sub_context)} chars.\nPlease solve this sub-problem."
        next_prompt = user_message

        # Reduced max steps for sub-problems
        sub_max_steps = min(5, self.max_steps - depth)

        for step in range(sub_max_steps):
            self.logger.info(f"[Depth {depth}] Step {step + 1}/{sub_max_steps}")

            try:
                response = sub_chat.send_message(next_prompt)
                response_text = response.text

                # Track usage
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    u = response.usage_metadata
                    self._update_stats(u.prompt_token_count, u.candidates_token_count)

                self.logger.info(f"[Depth {depth}] Thought: {response_text[:200]}...")

            except Exception as e:
                self.logger.error(f"[Depth {depth}] Error: {e}")
                return f"Error in recursive call: {e}"

            # Check for final answer
            if "FINAL ANSWER:" in response_text:
                answer = response_text.split("FINAL ANSWER:")[-1].strip()
                self.logger.info(f"[Depth {depth}] FINAL: {answer[:100]}...")
                return answer

            # Execute code if present
            code_blocks = self._extract_code_blocks(response_text)
            if code_blocks:
                full_code = "\n".join(code_blocks)
                execution_output = repl.execute(full_code)
                next_prompt = f"Code Output:\n{execution_output}\n\nNext step?"
            else:
                next_prompt = "Please provide code or FINAL ANSWER."

        self.logger.info(f"[Depth {depth}] Max sub-steps reached")
        return "Sub-problem reached max steps without answer."

    def _update_stats(self, input_tokens, output_tokens):
        if not input_tokens or not output_tokens:
            return
        self.stats["total_input_tokens"] += input_tokens
        self.stats["total_output_tokens"] += output_tokens
        self.stats["total_tokens"] += input_tokens + output_tokens

        # Calculate cost
        model = self.client.model_name
        pricing = self.PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens / 1000) * pricing["input"] + (
            output_tokens / 1000
        ) * pricing["output"]
        self.stats["estimated_cost"] += cost

    def _add_execution_node(self, depth: int, step: int, action: str, duration: float):
        """Add a node to the execution tree for visualization."""
        self.execution_tree.append(
            {"depth": depth, "step": step, "action": action, "duration": duration}
        )

    def _visualize_execution_tree(self) -> str:
        """Generate ASCII visualization of execution tree."""
        if not self.execution_tree:
            return "(No execution tree data)"

        lines = ["Execution Tree:"]
        for node in self.execution_tree:
            indent = "  " * node["depth"]
            prefix = "├─" if node["depth"] > 0 else ""
            lines.append(
                f"{indent}{prefix} Step {node['step']}: {node['action'][:40]}... ({node['duration']:.2f}s)"
            )

        return "\n".join(lines)

    def _finish_run(self):
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]

        self.logger.info("\n" + "=" * 50)
        self.logger.info("RLM Execution Complete")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Duration: {duration:.2f}s")
        self.logger.info(
            f"Total Tokens: {self.stats['total_tokens']:,} (In: {self.stats['total_input_tokens']:,}, Out: {self.stats['total_output_tokens']:,})"
        )
        self.logger.info(f"Estimated Cost: ${self.stats['estimated_cost']:.4f}")

        # Recursion stats
        rec_stats = self.recursion_guard.get_stats()
        self.logger.info(
            f"Recursion: {rec_stats['total_calls']} calls, max depth {rec_stats['max_depth_reached']}"
        )

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

        i = 0
        while i < len(lines):
            line = lines[i]

            if not in_code_block:
                # Look for start of python code block
                stripped = line.strip()
                if stripped.startswith("```python") or stripped == "```python":
                    in_code_block = True
                    current_block = []
                    i += 1
                    continue
            else:
                # We're inside a code block
                # Check if this line is ONLY ``` (the closing marker)
                stripped = line.strip()

                # The closing ``` should be on its own line (possibly with whitespace)
                if stripped == "```":
                    # End of code block
                    if current_block:
                        blocks.append("\n".join(current_block))
                    in_code_block = False
                    current_block = []
                else:
                    current_block.append(line)

            i += 1

        # Handle unclosed block (shouldn't happen, but just in case)
        if in_code_block and current_block:
            blocks.append("\n".join(current_block))

        return [b.strip() for b in blocks if b.strip()]
