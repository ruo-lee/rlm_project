"""
Baseline RLM Agent - Original paper implementation.

No caching, no parallel batch processing, no dynamic chunking.
Sequential llm_query calls only.
"""

from google.genai import types

from src.repl.baseline import BaselineREPL
from src.rlm.base import BaseRLMAgent
from src.rlm.config import SYSTEM_PROMPT_BASELINE


class BaselineRLMAgent(BaseRLMAgent):
    """
    Baseline RLM implementation matching the original paper.

    Features:
    - Sequential llm_query calls (no parallel)
    - No caching
    - No dynamic chunk sizing
    - RLM() recursive calls supported
    """

    VERSION = "Baseline"

    def __init__(self, output_dir: str = "."):
        super().__init__(output_dir)
        self.recursion_depth = 0
        self.max_recursion_depth = 5

    def _create_repl(self, context_text: str, depth: int = 0) -> BaselineREPL:
        """Create a baseline REPL with minimal features."""
        return BaselineREPL(
            context_str=context_text,
            llm_query_func=self._sub_llm_call,
            rlm_recursive_func=self._recursive_run,
            current_depth=depth,
        )

    def run(self, context_text: str, user_query: str) -> str:
        """Execute baseline RLM on the given context and query."""
        self._start_run(context_text, user_query)

        # Initialize REPL
        repl = self._create_repl(context_text, depth=0)

        # Initialize Chat Session
        self.chat = self.client.client.chats.create(
            model=self.client.model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_BASELINE
            ),
        )

        user_message = f"Query: {user_query}\n\nContext length: {len(context_text)} chars.\nPlease start by exploring the context."
        next_prompt = user_message

        for step in range(self.max_steps):
            self.logger.info(f"\n=== Step {step + 1}/{self.max_steps} ===")

            try:
                response = self.chat.send_message(next_prompt)
                response_text = response.text

                # Track usage
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    u = response.usage_metadata
                    self._update_stats(u.prompt_token_count, u.candidates_token_count)

                self.logger.info(f"[RLM Thought]:\n{response_text}")

            except Exception as e:
                self.logger.error(f"Error during LLM call: {e}")
                self._finish_run()
                return f"Error during execution: {e}"

            # Check for final answer
            if "FINAL ANSWER:" in response_text:
                self._finish_run()
                return response_text.split("FINAL ANSWER:")[-1].strip()

            # Extract and execute code
            code_blocks = self._extract_code_blocks(response_text)

            if code_blocks:
                full_code = "\n".join(code_blocks)
                self.logger.info(f"[Executing Code]:\n{full_code}")

                execution_output = repl.execute(full_code)
                self.logger.info(f"[Execution Output]:\n{execution_output}")

                next_prompt = f"Code Output:\n{execution_output}"
            else:
                next_prompt = "Please provide Python code to explore the context or give your FINAL ANSWER."

        self._finish_run()
        return "Max steps reached without final answer."

    def _sub_llm_call(self, prompt: str) -> str:
        """Sub-LLM call without caching."""
        self.logger.info(f"[Sub-LLM Call] Prompt: {prompt[:100]}...")
        result = self.client.generate_content(prompt)

        usage = result.get("usage", {})
        self._update_stats(
            usage.get("prompt_token_count", 0), usage.get("candidates_token_count", 0)
        )

        return result["text"]

    def _recursive_run(self, sub_context: str, query: str, depth: int) -> str:
        """Execute a recursive RLM call for sub-problem."""
        if depth > self.max_recursion_depth:
            return f"Max recursion depth ({self.max_recursion_depth}) exceeded"

        self.logger.info(f"\n{'='*20} RECURSIVE RLM (depth {depth}) {'='*20}")
        self.logger.info(f"Sub-query: {query}")
        self.logger.info(f"Sub-context length: {len(sub_context)} chars")

        # Create new REPL for recursive call
        repl = self._create_repl(sub_context, depth=depth)

        # Create new chat session
        sub_chat = self.client.client.chats.create(
            model=self.client.model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_BASELINE
            ),
        )

        user_message = f"Query: {query}\n\nContext length: {len(sub_context)} chars.\nPlease solve this sub-problem."
        next_prompt = user_message

        sub_max_steps = min(5, self.max_steps - depth)

        for step in range(sub_max_steps):
            self.logger.info(f"[Depth {depth}] Step {step + 1}/{sub_max_steps}")

            try:
                response = sub_chat.send_message(next_prompt)
                response_text = response.text

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    u = response.usage_metadata
                    self._update_stats(u.prompt_token_count, u.candidates_token_count)

                self.logger.info(f"[Depth {depth}] Thought: {response_text[:200]}...")

            except Exception as e:
                self.logger.error(f"[Depth {depth}] Error: {e}")
                return f"Error in recursive call: {e}"

            if "FINAL ANSWER:" in response_text:
                answer = response_text.split("FINAL ANSWER:")[-1].strip()
                self.logger.info(f"[Depth {depth}] FINAL: {answer[:100]}...")
                return answer

            code_blocks = self._extract_code_blocks(response_text)
            if code_blocks:
                full_code = "\n".join(code_blocks)
                execution_output = repl.execute(full_code)
                next_prompt = f"Code Output:\n{execution_output}\n\nNext step?"
            else:
                next_prompt = "Please provide code or FINAL ANSWER."

        self.logger.info(f"[Depth {depth}] Max sub-steps reached")
        return "Sub-problem reached max steps without answer."
