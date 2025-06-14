from models.base_model import BaseLM
from models.gemini_model_e import GeminiModelE
from models.gemini_model_j import GeminiModelJ
from models.openai_model_e import OpenAIModelE
from models.openai_model_j import OpenAIModelJ
from models.deepseek_model_e import DeepSeekModelE
from models.deepseek_model_j import DeepSeekModelJ
from models.groq_model_e import GroqModelE
from models.groq_model_j import GroqModelJ

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