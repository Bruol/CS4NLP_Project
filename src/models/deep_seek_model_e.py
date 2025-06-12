from src.models.base_model import ModelEBase, 
from src.config import DEEP_SEEK_API_KEY
from openai import OpenAI


class DeepSeekModelE(ModelEBase):
    """
    A Model-E implementation using the DeepSeek-R1 Model.
    """

    def __init__(self, model_name: str = "deepseek-ai/deepseek-reasoner"):
        super().__init__(model_name)
        client = OpenAI(api_key=DEEP_SEEK_API_KEY)
        self.model = client.models

    def generate_response(self, prompt: str) -> dict:
        messages = [{"role": "user", "content": prompt}]
        
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        thought = response.choices[0].message.reasoning_content
        response_text = response.choices[0].message.content

        return {
            "response": response_text,
            "label": 
            "thought": thought,
            "thought_steps": thought.split("\n\n")  # Assuming thought steps are separated by double newlines
        }
