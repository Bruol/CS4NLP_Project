from src.models.base_model import ModelEBase
from groq import Groq
from src.config import GROQ_API_KEY
from typing import Dict, Any


class GroqModelE(ModelEBase):
    """
    A Model-E implementation using Groq models.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(model_name)
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generates a response from the Groq model with chain-of-thought reasoning.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            Dict[str, Any]: A dictionary containing the response, response label, thought, and thought steps.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            reasoning_format="parsed"
        )

        print(response)

        response_text = response.choices[0].message.content
        reasoning = response.choices[0].message.reasoning
        
        # Split the response into thought steps (assuming they're separated by newlines)
        thought_steps = [step.strip() for step in response_text.split("\n\n") if step.strip()]

        return {
            "response": response_text,
            "response_label": self.parse_response(response_text),
            "thought": reasoning,
            "thought_steps": thought_steps
        } 