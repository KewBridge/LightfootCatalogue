# Python Modules
import os
from tqdm import tqdm

from typing import Optional, Union

# Import Custom Modules
from lightcat.models import get_model
from lightcat.utils.prompt_utils import PromptBuilder, PromptLoader
from lightcat.utils.save_utils import save_json, save_csv_from_json, verify_json
from lightcat.data_processing.text_processing import TextProcessor
# Logging
from lightcat.utils import get_logger
logger = get_logger(__name__)

class BaseModel:
    """Base class for all models in the pipeline."""

    TEMP_TEXT_FILE = "temp.txt"

    def __init__(self,
                 prompt: Union[Optional[str], PromptLoader] = None
                 ):
        """
        Base model encapsulating common model functionality.

        Parameters:
            prompt: The name of the prompt file, path to it, or PromptLoader instance
        """
        
        self.prompt = PromptLoader(prompt) if isinstance(prompt, str) else prompt
        self.promptBuilder = PromptBuilder(self.prompt)
        self.model_name = self.prompt.get("model", "qwen2.5")
        self.batch_size = self.prompt.get("batch_size", 1)
        self.max_new_tokens = self.prompt.get("max_tokens", 4096)
        self.temperature = 0.1
        self.save_path = self.prompt.get("output_save_path", "./outputs/default/")

        if not(os.path.isdir(self.save_path)):
            os.makedirs(self.save_path)

        # Define a temporary file to store extracted text
        self.TEMP_TEXT_FILE = os.path.join(self.save_path, self.TEMP_TEXT_FILE)

        # Load the model, prompt, and the conversation
        self.model = None
    
    def load_model(self, model_name: Optional[str] = None) -> object:
        
        model_name = model_name or self.model_name
        logger.info(f"Loading model: {model_name} with batch size: {self.batch_size}, max tokens: {self.max_new_tokens}, temperature: {self.temperature}")
        return get_model(model_name)(self.batch_size, self.max_new_tokens, self.temperature)


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