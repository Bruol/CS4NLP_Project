from models.base_model import ModelEBase
from pydantic import BaseModel
from openai import OpenAI
from config import OPENAI_API_KEY

REASONING_EFFORTS = ["low", "medium", "high"]

class OpenAIModelEResponse(BaseModel):
    response: str
    thought_steps: list[str]

class OpenAIModelE(ModelEBase):
    """
    A Model-E implementation using the DeepSeek-R1 Model.
    """

    def __init__(self, model_name: str, reasoning_effort: str = "medium"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort if reasoning_effort in REASONING_EFFORTS else "medium"

    def generate_response(self, prompt: str) -> dict:
        
        response = self.client.responses.parse(
            model=self.model_name,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
            text_format=OpenAIModelEResponse
        )

        thought_steps = response.output_parsed.thought_steps
        response_text = response.output_parsed.response

        return {
            "response": response_text,
            "response_label": self.parse_response(response_text),
            "thought_steps": thought_steps,
            "thought": "\n".join(thought_steps)
        }
