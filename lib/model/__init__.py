import importlib

MODELS = {
    "qwen_model": "lib.model.qwen_model"
}


def get_model(model_name: str):

    models = MODELS

    try:
        module = importlib.import_module(models[model_name])
        model = getattr(module, "QWEN_Model")
        return model
    except:
        raise ImportError(f"{model_name} cannot be imported. Provide the correct name of the model. \n==> {models.keys()}")