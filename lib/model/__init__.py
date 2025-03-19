import importlib

MODELS = {
    "qwen_model": ("lib.model.qwen_model", "QWEN_Model")
}


def get_model(model_name: str):

    models = MODELS

    try:
        module = importlib.import_module(models[model_name][0])
        model = getattr(module, models[model_name][1])
        return model
    except:
        raise ImportError(f"{model_name} cannot be imported. Provide the correct name of the model. \n==> {models.keys()}")