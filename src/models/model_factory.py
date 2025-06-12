from src.models.base_model import BaseLM
from src.models.gemini_model_e import GeminiModelE
from src.models.gemini_model_j import GeminiModelJ

SUPPORTED_MODELS = {
    "google/gemini-2.5-flash": "gemini-2.5-flash-preview-05-20",
    "google/gemini-2.5-pro": "gemini-2.5-pro-preview-06-05"
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
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    return model_class(model_name=SUPPORTED_MODELS[model_name]) 