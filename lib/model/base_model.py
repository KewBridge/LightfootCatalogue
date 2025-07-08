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
        self.model_name = prompt["model"]
        self.batch_size = prompt["batch_size"] if prompt["batch_size"] is not None else 3
        self.max_new_tokens = prompt["max_tokens"] if prompt["max_tokens"] is not None else 4096
        self.temperature = 0.1
        self.save_path = prompt["output_save_path"] if prompt["output_save_path"] is not None else "./outputs/default/"
        

        if not(os.path.isdir(self.save_path)):
            os.makedirs(self.save_path)

        # Defining a temperory file to store extracted text
        self.TEMP_TEXT_FILE = os.path.join(self.save_path, self.TEMP_TEXT_FILE)

        self.timeout = prompt["timeout"] if prompt["timeout"] is not None else 4
        
        # Load the model, prompt, and the conversation
        self.model = None
    
    def load_model(self, model_name: Optional[str] = None) -> object:
        
        model_name = model_name if model_name is not None else self.model_name
        return get_model(model_name)(None, self.batch_size, self.max_new_tokens, self.temperature)

    def info(self) -> str:
        """
        Info on the the model pipeline and the paramters used

        Returns:
            message (str): brief information of parameters and model name
        """
        message = f"Model: {self.model_name} | Batch Size: {self.batch_size}, Max Tokens: {self.max_new_tokens}, Temperature: {self.temperature}"

        print(message)
    

    def _save_to_file(self, file: str, text: str, mode: str="w") -> None:
        """
        (Private function)
        Saves the extracted text to the file
        """

        with open(file, mode) as f:
            f.write(text)
    

    def _load_from_file(self, file: str) -> str:
        """
        (Private function)
        Load the extracted text from the file

        Returns:
            text: the text read from the file

        """

        text = ""

        with open(file, "r") as f:
            text = f.read()
        
        return text


    def _get_save_file_name(self, save_file_name: str) -> str:
        """
        (Private function)
        Get the ideal name for the save file. This function checks for any duplicates and adds version numbers
        to the end of given save file names to create unique save file names.
        This ensures no overwriting

        Parameters:
            save_file_name (str): the input name for the save file as given by user.

        Returns:
            final_save_file_name (str): the finalised save file name
        """

        # Load all files under save path as a hashset
        files = {file for file in os.listdir(self.save_path) if os.path.isfile(os.path.join(self.save_path, file))}
        
        id = 0
        final_save_file_name = save_file_name

        # Do a conditional loop to find best name for save file
        while f"{final_save_file_name}.json" in files:

            final_save_file_name = f"{save_file_name}_{id}"
            id += 1

        return final_save_file_name


    def __call__(self):
        pass
