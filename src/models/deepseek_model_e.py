from src.models.base_model import ModelEBase
from openai import OpenAI
from src.config import DEEP_SEEK_API_KEY

class DeepSeekModelE(ModelEBase):
    """
    A Model-E implementation using the DeepSeek-R1 Model.
    """

    def __init__(self, model_name: str = "deepseek-reasoner", base_url: str = "https://api.deepseek.com"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=DEEP_SEEK_API_KEY, base_url=base_url)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> dict:
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )


        thought = response.choices[0].message.reasoning_content
        response_text = response.choices[0].message.content

        return {
            "response": response_text,
            "response_label": self.parse_response(response_text),
            "thought": thought,
            "thought_steps": thought.split("\n\n")
        }
