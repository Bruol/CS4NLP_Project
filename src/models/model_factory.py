from src.models.base_model import BaseLM
from src.models.gemini_model_e import GeminiModelE
from src.models.gemini_model_j import GeminiModelJ
from src.models.openai_model_e import OpenAIModelE
from src.models.openai_model_j import OpenAIModelJ
from src.models.deepseek_model_e import DeepSeekModelE
from src.models.deepseek_model_j import DeepSeekModelJ
from src.models.groq_model_e import GroqModelE
from src.models.groq_model_j import GroqModelJ
from src.config import DEEP_SEEK_API_KEY
from src.config import GROQ_API_KEY
from src.config import OPENAI_API_KEY

SUPPORTED_MODELS = {
    "google/gemini-2.5-flash": "gemini-2.5-flash-preview-05-20",
    "google/gemini-2.5-pro": "gemini-2.5-pro-preview-06-05",
    "deepseek-ai/deepseek-r1": "deepseek-reasoner",
    "groq/deepseek-r1-distill-llama": "deepseek-r1-distill-llama-70b",
    "groq/llama-3.3": "llama-3.3-70b-versatile",
    "openai/o4-mini": "o4-mini",
    "openai/gpt-4o": "gpt-4o"
}

def get_model(model_name: str, model_type: str) -> BaseLM:
    """
    Factory function to get a model instance.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    provider = model_name.split("/")[0]

    if provider == "google":
        if model_type == "e":
            model_class = GeminiModelE
        elif model_type == "j":
            model_class = GeminiModelJ
    elif provider == "deepseek-ai":
        if model_type == "e":
            model_class = DeepSeekModelE
        elif model_type == "j":
            model_class = DeepSeekModelJ
    elif provider == "groq":
        if model_type == "e":
            model_class = GroqModelE
        elif model_type == "j":
            model_class = GroqModelJ
    elif provider == "openai":
        if model_type == "e":
            model_class = OpenAIModelE
        elif model_type == "j":
            model_class = OpenAIModelJ
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model_class(model_name=SUPPORTED_MODELS[model_name]) 