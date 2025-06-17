from google.genai import Client, types
from models.base_model import ModelEBase
from config import GOOGLE_API_KEY
from typing import Dict, Any



class GeminiModelE(ModelEBase):
    """
    A Model-E implementation using the Gemini 1.5 Flash model.
    """

    def __init__(self, model_name: str = "models/gemini-2.5-flash-preview-05-20"):
        super().__init__(model_name)
        client = Client(api_key=GOOGLE_API_KEY)
        self.model = client.models
        self.model_name = model_name
        self.thinking_budget = 8000

    def set_reasoning_effort(self, reasoning_effort: str):
        """
        Sets the reasoning effort for the model.
        """
        self.thinking_budget = 8000 if reasoning_effort == "medium" else 16000 if reasoning_effort == "high" else 4000

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generates a response from the Gemini model.
        If with_cot is True, it prefixes the prompt to encourage a chain-of-thought response.

        Args:
            prompt (str): The input prompt for the model.
            with_cot (bool): Whether to generate a chain-of-thought response.

        Returns:
            str: The model's response.
        """
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True
            )
        )

        response = self.model.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        try:
            thought_candidate = [part for candidate in response.candidates for part in candidate.content.parts if part.thought]
            thought = thought_candidate[0].text
            thought_steps = thought.split("\n\n") 
            answer_label = self.parse_response(response.text)
            return {
                "response": response.text,
                "response_label": answer_label,
                "thought": thought,
                "thought_steps": thought_steps
            }
        except IndexError:
            raise IndexError("No thought found in the response")
