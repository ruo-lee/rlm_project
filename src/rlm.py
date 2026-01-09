import re

from google.genai import types
from termcolor import colored

from src.llm_client import GeminiClient
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
        self.chat_history = (
            []
        )  # Use a simple list for history management with google-genai chats
        self.chat = None  # Will store the chat session

    def run(self, context_text: str, user_query: str):
        print(colored(f"--- Starting RLM on query: {user_query} ---", "green"))

        # 1. Initialize REPL
        repl = PythonREPL(context_text, self._sub_llm_call)

        # 2. Initialize Chat Session with System Prompt
        # google-genai supports chats.
        # We start a chat with the system instruction config.
        self.chat = self.client.client.chats.create(
            model=self.client.model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_TEMPLATE
            ),
        )

        # Initial user message
        user_message = f"Query: {user_query}\n\nContext length: {len(context_text)} chars.\nPlease start by exploring the context."

        print(colored("RLM Initialized. Entering loop...", "cyan"))

        next_prompt = user_message

        for step in range(self.max_steps):
            print(colored(f"\n--- Step {step + 1}/{self.max_steps} ---", "yellow"))

            # Send message to chat
            try:
                # Note: google-genai Chat.send_message might not support stream argument directly in this version.
                # Switching to synchronous call for stability.
                print(colored("[RLM Thought] (Thinking...):", "blue"))
                response = self.chat.send_message(next_prompt)
                response_text = response.text
                print(colored(response_text, "blue"))
            except Exception as e:
                print(colored(f"Error during LLM call: {e}", "red"))
                return "Error during execution."

            # print(colored(f"[RLM Thought]:\n{response_text}", "blue")) # Already printed via stream

            # Parse code
            code_blocks = self._extract_code_blocks(response_text)

            if not code_blocks:
                if "FINAL ANSWER:" in response_text:
                    return response_text.split("FINAL ANSWER:")[-1].strip()

                print(
                    colored(
                        "[System]: No code block found. Asking model to write code or finish.",
                        "red",
                    )
                )
                next_prompt = "You didn't provide any code. Please write Python code to inspect the context or output 'FINAL ANSWER:'."
                continue

            # Execute code
            full_code = "\n".join(code_blocks)
            print(colored(f"[Executing Code]:\n{full_code}", "magenta"))

            execution_output = repl.execute(full_code)
            # Truncate output to avoid blowing up context too fast
            msg_output = (
                execution_output[:2000] + "...(truncated)"
                if len(execution_output) > 2000
                else execution_output
            )
            print(colored(f"[Execution Output]:\n{msg_output}", "white"))

            # Pass output back to LLM in next turn
            next_prompt = f"Code Output:\n{execution_output}\n\nBased on this, what is the next step?"

            if "FINAL ANSWER:" in response_text:
                # It might have output the answer AND code.
                # Usually we trust the text if it says final answer.
                return response_text.split("FINAL ANSWER:")[-1].strip()

        return "Max steps reached without final answer."

    def _sub_llm_call(self, prompt: str) -> str:
        """This is the function available inside REPL as `llm_query`."""
        return self.client.generate_content(prompt)

    def _extract_code_blocks(self, text: str) -> list[str]:
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]
