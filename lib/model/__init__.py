import importlib

MODELS = {
    "qwen2": ("lib.model.hf_models.qwen_model", "QWEN2_VL_Model"),
    "qwen2.5": ("lib.model.hf_models.qwen_model", "QWEN2_5_VL_Model")
}


def get_model(model_name: str):

    models = MODELS

    try:
        module = importlib.import_module(models[model_name][0])
        model = getattr(module, models[model_name][1])
        return model
    except:
        raise ImportError(f"{model_name} cannot be imported. Provide the correct name of the model. \n==> {models.keys()}")