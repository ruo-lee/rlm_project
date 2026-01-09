import time

from google.genai import types
from termcolor import colored

from src.llm_client import GeminiClient
from src.logger_config import setup_logger
from src.repl_optimized import PythonREPL, RecursionGuard
from src.rlm.config import PRICING, SYSTEM_PROMPT_OPTIMIZED


class RLMAgent:
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
            "llm_calls": 0,
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
                system_instruction=SYSTEM_PROMPT_OPTIMIZED
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

                # Count this as one LLM call after streaming completes
                self.stats["llm_calls"] += 1
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
            usage.get("prompt_token_count", 0),
            usage.get("candidates_token_count", 0),
            is_new_call=True,
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
                system_instruction=SYSTEM_PROMPT_OPTIMIZED
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

    def _update_stats(self, input_tokens, output_tokens, is_new_call=False):
        if not input_tokens or not output_tokens:
            return
        self.stats["total_input_tokens"] += input_tokens
        self.stats["total_output_tokens"] += output_tokens
        self.stats["total_tokens"] += input_tokens + output_tokens

        # Only increment llm_calls when explicitly a new call
        if is_new_call:
            self.stats["llm_calls"] += 1

        # Calculate cost
        model = self.client.model_name
        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
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
