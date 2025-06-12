from google.genai import Client, types
from src.config import GOOGLE_API_KEY
import json



model_name = "models/gemini-2.5-flash-preview-05-20"
client = Client(api_key=GOOGLE_API_KEY)

prompt = "What is the capital of France?"


response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=8000,
            include_thoughts=True
        )
    )
)

json

print(response)