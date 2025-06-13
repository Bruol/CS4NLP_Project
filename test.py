from google.genai import Client, types
from config import GOOGLE_API_KEY, DEEP_SEEK_API_KEY, OPENAI_API_KEY
from openai import OpenAI
import json
from pydantic import BaseModel


# model_name = "models/gemini-2.5-flash-preview-05-20"
# client = Client(api_key=GOOGLE_API_KEY)


# prompt = "What is the capital of France?"


# response = client.models.generate_content(
#     model=model_name,
#     contents=prompt,
#     config=types.GenerateContentConfig(
#         thinking_config=types.ThinkingConfig(
#             thinking_budget=8000,
#             include_thoughts=True
#         )
#     )
# )

# print("response: ", response)
# thought_candidate = [part for candidate in response.candidates for part in candidate.content.parts if part.thought]
# thought = thought_candidate[0].text

client = OpenAI(api_key=OPENAI_API_KEY)

class OpenAIModelEResponse(BaseModel):
    response: str
    thought_steps: list[str]

response = client.responses.parse(
    model="o4-mini",
    input="What is the capital of France?",
    reasoning={"effort": "medium"},
    text_format=OpenAIModelEResponse
)

print("response: ", response)