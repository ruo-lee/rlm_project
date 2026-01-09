import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

class GeminiClient:
    def __init__(self, model_name=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env.local")
        
        self.client = genai.Client(api_key=api_key)
        # Use provided model_name, or env var, or fallback default
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME") or "gemini-3-pro-preview"

    def generate_content(self, prompt, system_instruction=None):
        try:
            config = None
            if system_instruction:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error: {e}"
