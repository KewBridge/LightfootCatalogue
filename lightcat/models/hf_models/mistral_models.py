
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Custom Modules
from lightcat.models.hf_models.hf_model import HF_Model

# Logging
from lightcat.utils import get_logger
logger = get_logger(__name__)

class MISTRAL_7B_INSTRUCT(HF_Model):

    DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    MODEL_TYPE = "single" # multi if multi-model, single if single model => Used to define the type of prompt to be used
    CONTEXT_LENGTH = 8192 # Default context length for the model

    def __init__(self,
                 batch_size: int = 1, # Batch size for inference
                 max_new_tokens: int = 4096, # Maximum number of tokens
                 temperature: float = 0.1, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
                ):
        """
        Mistral 7B Instruct model class

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


        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,torch_dtype=torch.bfloat16, device_map="auto"
        )

        self.device = model.device
        self._load_context_length()
    
        return model    

    def _load_processor(self) -> object:
        """
        Loads the pre-processor that is used to pre-process the input prompt and images.
    
        Return:
            processor (object): Returns the loaded pretrained processor for the model.
        """

        processor = AutoTokenizer.from_pretrained(self.model_name)
        processor.padding_side = "left"
    
        return processor