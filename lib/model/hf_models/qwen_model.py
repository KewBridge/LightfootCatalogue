# Python Modules

from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

# Custom Modules
from lib.model.hf_models.hf_model import HF_Model


# Logging
from lib.utils import get_logger
logger = get_logger(__name__)

class QWEN2_VL_Model(HF_Model):

    DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME, # Model name
                 batch_size: int = 1, # Batch size for inference
                 max_new_tokens: int = 4096, # Maximum number of tokens
                 temperature: float = 0.3, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
                ):
        """
        QWEN model class

        This class loads the necessary modules and performs inference given conversation and input

        Parameters:
            batch_size (int): batch size for inference
            max_new_tokens (int): Maximum number of tokens
            temperature (float): Model temperature. 0 to 2. Higher the value the more random and
                                 lower the temperature the more focussed and deterministic.
        """
        super().__init__(model_name, batch_size, max_new_tokens, temperature)


    def _load_model(self) -> object:
        """
        Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """


        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )

        self.device = model.device
        # Enable gradient checkpointing to reduce memory usage
        model.gradient_checkpointing_enable()
    
        return model    

    # def __call__(self, conversation: list, images:list[str]=None, 
    #              debug: bool=False, max_new_tokens=None,
    #              add_padding: bool=True, skip_special_tokens: bool=True) -> list:
    #     """
    #     Performs inference on the given set of images and/or text.

    #     When images are provided, the text is extracted.
    #     When text is provided, images is set to None and inference is determined by conversation
    
    #     Parameters:
    #         conversation (list): The input prompt to the model
    #         images (list): A set of images to batch inference.
    #         debug (bool): Used to print debug prompts
    #         max_new_tokens (int): The maximum number of new tokens to generate. If None, the default max_new_tokens is used.
    #         add_padding (bool): Whether to add padding to the input text. Default is True.
    #         skip_special_tokens (bool): Whether to skip special tokens in the output. Default is True.

    #     Return:
    #         output_text (list): A set of model outputs for given set of images.
    #     """

    #     self._check()

    #     if max_new_tokens is None:
    #         max_new_tokens = self.max_new_tokens    
    #     # Set the device to the model's device
    #     self.eval()

    #     # Process the input conversation
    #     if debug:
    #         logger.debug("\tProcessing inputs...")
    #     inputs = self.process_chat_inputs(conversation, images, add_padding=add_padding) 
        

    #     output_text = self.inference_chat_model(inputs, max_new_tokens, skip_special_tokens=skip_special_tokens)
        


    #     return output_text

    

class QWEN2_5_VL_Model(QWEN2_VL_Model):

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, model_name = DEFAULT_MODEL_NAME, batch_size = 1, max_new_tokens = 4096, temperature = 0.3):
        super().__init__(model_name, batch_size, max_new_tokens, temperature)

    def _load_model(self):
        """
        Load the Qwen2.5-VL pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,temperature=self.temperature, torch_dtype="auto", device_map="auto"
        )

        self.device = model.device
        # Enable gradient checkpointing to reduce memory usage
        model.gradient_checkpointing_enable()
    
        return model 