from src.models.base_model import BaseLM
from src.models.gemini_model_e import GeminiModelE
from src.models.gemini_model_j import GeminiModelJ

SUPPORTED_MODELS = {
    "models/gemini-2.5-flash-preview-05-20",
}

def get_model(model_name: str, model_type: str) -> BaseLM:
    """
    Factory function to get a model instance.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    if model_type == "e":
        model_class = GeminiModelE
    elif model_type == "j":
        model_class = GeminiModelJ
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    
    return model_class(model_name=model_name) 