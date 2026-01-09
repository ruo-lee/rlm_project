import re
import time

from google.genai import types
from termcolor import colored

from src.llm_client import GeminiClient
from src.logger_config import setup_logger
from src.repl import PythonREPL

SYSTEM_PROMPT_TEMPLATE = """
You are a Recursive Language Model (RLM).
You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a Python REPL environment.

The REPL environment is initialized with:
1. `context`: A string variable containing the text you need to process.
2. `llm_query(prompt)`: A function that allows you to query a sub-LLM. 
   - Use this to summarize chunks, extract specific information, or answer questions about parts of the context.
   - The sub-LLM sees ONLY the prompt you pass to it, NOT the full original context (unless you pass parts of it).
3. `print()`: Use this to see the output of your code.

Process:
1. EXPLORE: Check the length of `context`, peek at the beginning/end, or search for keywords using Python code.
2. PLAN: Decide how to break down the problem.
3. EXECUTE: Write Python code to chunk the context and call `llm_query` on chunks if the text is too long or complex.
4. SYNTHESIZE: Gather results from your sub-calls and print them.
5. ANSWER: When you have the answer, print "FINAL ANSWER: [your answer]" to finish.

CRITICAL INSTRUCTIONS:
- You are running in a loop. You write code -> It executes -> You see the output -> You write more code.
- ALWAYS wrap your Python code in ```python ... ``` blocks.
- DO NOT just guess. Use the `context` variable.
- To finish, you MUST print a line starting with "FINAL ANSWER:".
"""


class RLMAgent:
    def __init__(self, output_dir: str = "."):
        self.client = GeminiClient()
        self.max_steps = 10
        self.chat_history = []
        self.chat = None

        # Metrics
        self.stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "start_time": 0,
            "end_time": 0,
            "steps": [],
        }

        # Setup Logger
        self.logger, self.log_file = setup_logger()

    def run(self, context_text: str, user_query: str):
        self.stats["start_time"] = time.time()
        self.logger.info("--- Starting RLM ---")
        self.logger.info(f"Query: {user_query}")
        self.logger.info(f"Context Length: {len(context_text)} chars")

        # 1. Initialize REPL
        repl = PythonREPL(context_text, self._sub_llm_call)

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

    def _update_stats(self, input_tokens, output_tokens):
        if not input_tokens or not output_tokens:
            return
        self.stats["total_input_tokens"] += input_tokens
        self.stats["total_output_tokens"] += output_tokens
        self.stats["total_tokens"] += input_tokens + output_tokens

    def _finish_run(self):
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]
        self.logger.info("\n--- RLM Execution Copmlete ---")
        self.logger.info(f"Total Duration: {duration:.2f}s")
        self.logger.info(
            f"Total Tokens: {self.stats['total_tokens']} (In: {self.stats['total_input_tokens']}, Out: {self.stats['total_output_tokens']})"
        )
        self.logger.info(f"Full log saved to: {self.log_file}")

    def _extract_code_blocks(self, text: str) -> list[str]:
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]
