import importlib
from lightcat.utils import get_logger

logger = get_logger(__name__)
HF_MODEL_PATH = "lightcat.models.hf_models"
GROQ_MODEL_PATH = "lightcat.models.groq_models"
MODELS = {
    # Hugging Face Models
    "qwen2": (f"{HF_MODEL_PATH}.qwen_models", "QWEN2_VL_Model"),
    "qwen2.5": (f"{HF_MODEL_PATH}.qwen_models", "QWEN2_5_VL_Model"),

    # Hugging Face OCR Models
    "mistral7b": (f"{HF_MODEL_PATH}.mistral_models", "MISTRAL_7B_INSTRUCT"),

    # Groq Models
    "gpt-oss-20b": (f"{GROQ_MODEL_PATH}.groq_model", "Groq_GPT_OSS_20B_Model"),
    "gpt-oss-120b": (f"{GROQ_MODEL_PATH}.groq_model", "Groq_GPT_OSS_120B_Model"),
    "gpt-oss-safeguard-20b": (f"{GROQ_MODEL_PATH}.groq_model", "Groq_GPT_OSS_Safeguard_20B_Model"),
    "llama4-maverick-17b": (f"{GROQ_MODEL_PATH}.groq_model", "LLAMA4_17B_Maverick_Model"),
    "llama4-scout-17b": (f"{GROQ_MODEL_PATH}.groq_model", "LLAMA4_17B_Scout_Model"),
}


def get_model(model_name: str):
    """Get model class by name.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Model class
        
    Raises:
        KeyError: If model_name not found in MODELS
        ImportError: If module cannot be imported or class not found
    """
    models = MODELS

    try:
        module_path, class_name = models[model_name]
    except KeyError:
        available = ", ".join(models.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: {available}")
    
    try:
        module = importlib.import_module(module_path)
        logger.info(f"Importing model: {model_name}")
        model = getattr(module, class_name)
        logger.info(f"Model imported successfully: {model_name}")
        return model
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot import model '{model_name}' from {module_path}.{class_name}: {e}")