# Python Modules
import os
from tqdm import tqdm

from typing import Optional, Union

# Import Custom Modules
from lib.model import get_model
from lib.utils.promptLoader import PromptLoader
from lib.utils.save_utils import save_json, save_csv_from_json, verify_json
from lib.data_processing.text_processing import TextProcessor
# Logging
from lib.utils import get_logger
logger = get_logger(__name__)

class BaseModel:

    TEMP_TEXT_FILE = "temp.txt"
    SAVE_TEXT_INTERVAL = 3

    def __init__(self,
                 prompt: Union[Optional[str], PromptLoader] = None,
                 **kwargs
                 ):
        """
        Base model encapsulating the available models

        Parameters:
            model_name (str): the name of the model
            prompt (str): The name of the prompt file or the path to it
            batch_size (int): Batch size for inference
            max_new_tokens (int): Maximum number of tokens
            temperature (float): Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
            save_path (str): Where to save the outputs
            timeout (int): The number of times to rechech for JSON validation (currrently a placeholder)
            **kwargs (dict): extra parameters for other models
        """
        
        self.prompt = PromptLoader(prompt) if isinstance(prompt, str) else prompt
        self.model_name = self.prompt.get("model", "qwen2.5")
        self.batch_size = self.prompt.get("batch_size", 1)
        self.max_new_tokens = self.prompt.get("max_tokens", 4096)
        self.temperature = 0.1
        self.save_path = self.prompt.get("output_save_path", "./outputs/default/")

        if not(os.path.isdir(self.save_path)):
            os.makedirs(self.save_path)

        # Defining a temperory file to store extracted text
        self.TEMP_TEXT_FILE = os.path.join(self.save_path, self.TEMP_TEXT_FILE)

        self.timeout = self.prompt.get("timeout", 4)

        # Load the model, prompt, and the conversation
        self.model = None
    
    def load_model(self, model_name: Optional[str] = None) -> object:
        
        model_name = model_name or self.model_name
        logger.info(f"Loading model: {model_name} with batch size: {self.batch_size}, max tokens: {self.max_new_tokens}, temperature: {self.temperature}")
        return get_model(model_name)(None, self.batch_size, self.max_new_tokens, self.temperature)


    def info(self) -> str:
        """
        Info on the the model pipeline and the paramters used

        Returns:
            message (str): brief information of parameters and model name
        """
        message = f"Model: {self.model_name} | Batch Size: {self.batch_size}, Max Tokens: {self.max_new_tokens}, Temperature: {self.temperature}"

        print(message)

    def __call__(self):
        pass
