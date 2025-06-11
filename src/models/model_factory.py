from src.models.base_model import BaseLM
from src.models.gemini_model_e import GeminiModelE
from src.models.gemini_model_j import GeminiModelJ

SUPPORTED_MODELS = {
    "gemini-2.5-flash": GeminiModelE,
    "gemini-2.5-pro": GeminiModelJ,
}

def get_model(model_name: str) -> BaseLM:
    """
    Factory function to get a model instance.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    model_class = SUPPORTED_MODELS[model_name]
    return model_class(model_name=model_name) 