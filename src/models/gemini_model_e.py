from google.genai import Client
from src.models.base_model import ModelEBase
from src.config import GOOGLE_API_KEY



class GeminiModelE(ModelEBase):
    """
    A Model-E implementation using the Gemini 1.5 Flash model.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__(model_name)
        client = Client(api_key=GOOGLE_API_KEY)
        self.model = client.models.get(model_name)

    def generate_response(self, prompt: str, with_cot: bool = False) -> str:
        """
        Generates a response from the Gemini model.
        If with_cot is True, it prefixes the prompt to encourage a chain-of-thought response.

        Args:
            prompt (str): The input prompt for the model.
            with_cot (bool): Whether to generate a chain-of-thought response.

        Returns:
            str: The model's response.
        """
        if with_cot:
            prompt = "Think step-by-step and then answer the following question. \n\n" + prompt
        
        response = self.model.generate_content(prompt)
        return response.text 