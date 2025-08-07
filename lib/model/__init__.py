import importlib
from lib.utils import get_logger

logger = get_logger(__name__)

MODELS = {
    "qwen2": ("lib.model.hf_models.qwen_model", "QWEN2_VL_Model"),
    "qwen2.5": ("lib.model.hf_models.qwen_model", "QWEN2_5_VL_Model"),


    # OCR Models
    "mistral7b": ("lib.model.hf_models.mistral_model", "MISTRAL_7B_INSTRUCT"),
}


def get_model(model_name: str):

    models = MODELS

    try:
        module = importlib.import_module(models[model_name][0])
        logger.info(f"Importing model: {model_name}")
        model = getattr(module, models[model_name][1])
        logger.info(f"Model imported successfully: {model_name}")
        return model
    except:
        raise ImportError(f"{model_name} cannot be imported. Provide the correct name of the model. \n==> {models.keys()}")