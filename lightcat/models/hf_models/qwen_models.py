# Python Modules

from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

# Custom Modules
from lightcat.models.hf_models.hf_model import HF_Model


# Logging
from lightcat.utils import get_logger
logger = get_logger(__name__)

class QWEN2_VL_Model(HF_Model):

    DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    MODEL_TYPE = "multi" # multi if multi-model, single if single model => Used to define the type of prompt to be used
    CONTEXT_LENGTH = 32768 # Default context length for the model

    def __init__(self,
                 batch_size: int = 1, # Batch size for inference
                 max_new_tokens: int = 4096, # Maximum number of tokens
                 temperature: float = 0.3, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
                ):
        """
        QWEN2 VL model class

        This class loads the necessary modules and performs inference given conversation and input

        Parameters:
            batch_size (int): batch size for inference
            max_new_tokens (int): Maximum number of tokens
            temperature (float): Model temperature. 0 to 2. Higher the value the more random and
                                 lower the temperature the more focussed and deterministic.
        """
        super().__init__(self.DEFAULT_MODEL_NAME, batch_size, max_new_tokens, temperature)


    def _load_model(self) -> object:
        """
        Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """


        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )


        self._load_context_length()
    
        return model    

    

class QWEN2_5_VL_Model(QWEN2_VL_Model):

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    MODEL_TYPE = "multi" # multi if multi-model, single if single model => Used to define the type of prompt to be used
    CONTEXT_LENGTH = 32768 # Default context length for the model

    def __init__(self, batch_size = 1, max_new_tokens = 4096, temperature = 0.3):
        super().__init__(batch_size, max_new_tokens, temperature)

    def _load_model(self):
        """
        Load the Qwen2.5-VL pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )

        self._load_context_length()
    
        return model 