import os
from tqdm import tqdm
from lib.model import get_model
from lib.utils.promptLoader import PromptLoader
from lib.utils.utils import debugPrint
from lib.utils.json_utils import save_jsons

from lib.utils.text_utils import convertToTextBlocks


class BaseModel:

    TEMP_TEXT_FILE = "./temp.txt"
    SAVE_TEXT_INTERVAL = 3

    def __init__(self, 
                 model_name: str,
                 prompt: str = None,
                 conversation: list = None,
                 batch_size: int = 3, # Batch size for inference
                 max_new_tokens: int = 5000, # Maximum number of tokens
                 temperature: float = 0.2, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
                 save_path: str = None, # Where to save the outputs
                 **kwargs
                 ):
        """
        Base model encapsulating the available models

        Parameters:
            model_name (str): the name of the model
            prompt (str): The name of the prompt file or the path to it
            conversation (list): Input conversation into the model
            batch_size (int): Batch size for inference
            max_new_tokens (int): Maximum number of tokens
            temperature (float): Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
            save_path (str): Where to save the outputs
            **kwargs (dict): extra parameters for other models
        """
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.save_path = save_path
        self.model = get_model(self.model_name)(self.batch_size, self.max_new_tokens, self.temperature, **kwargs)

        self.prompt = PromptLoader(prompt)#prompt
        self.conversation = self.prompt.get_conversation() if conversation is None else conversation

    def info(self):
        message = f"Model: {self.model_name} | Batch Size: {self.batch_size}, Max Tokens: {self.max_new_tokens}, Temperature: {self.temperature}"

        print(message)
    

    def setNewPrompt(self, prompt: str, conversation: list=None):

        """
        Load a new prompt

        Parameters:
            prompt (str): The name of the prompt file or the path to it
            conversation (list): Input conversation into the model
        """

        if not(prompt is None) and conversation is None:
            self.prompt = PromptLoader(prompt)
            self.conversation = self.prompt.get_conversation()
        elif prompt is None and not(conversation is None):
            self.conversation = conversation
        else:
            raise ValueError("Received None for prompt and None for conversation")

    def _save_to_temp(self, text):

        with open(self.TEMP_TEXT_FILE, "w") as f:
            f.write(text)
    

    def _load_from_temp(self):

        text = ""

        with open(self.TEMP_TEXT_FILE, "r") as f:
            text = f.read()
        
        return text
    

    def extract_text(self, images: list, debug: bool = False):
        joined_text = "\n\n"

        debugPrint("Batching Images...", debug)

        # Create batches of images
        batched_images = [images[x:min(x + self.batch_size, len(images))] for x in range(0, len(images), self.batch_size)]

        # Add tqdm for progress tracking
        for ind, batch in enumerate(tqdm(batched_images, desc="Processing Batches", unit="batch")):
            #print(f">>> Batch {ind + 1} starting...")

            debugPrint("Extracting text from image", debug)
            image_conversation = self.prompt.getImagePrompt()
            extracted_text = self.model(image_conversation, batch, debug)

            debugPrint("\tJoining Outputs...", debug)
            # Join all the text and append it together with previous batches
            joined_text += "\n\n".join(extracted_text)

            if (ind + 1) % self.SAVE_TEXT_INTERVAL == 0:
                debugPrint("\tStoring at interval...", debug)
                self._save_to_temp(joined_text)

            debugPrint("\tBatch Finished", debug)

        return joined_text


    def __call__(self, images: list, debug: bool = False, save: bool = False, text=None) -> list:
        # TODO: Write the docu message

        self.info()

        if text is None:
            print("Extracting Text from Images")
            extracted_text = self.extract_text(images, debug)
        else:
            print("Skipping extraction...")
            print("Loading text from provided extracted text file")
            with open(text, "r") as file_:
                extracted_text = file_.read()

        print("Converting extracted text into Text Blocks")
        text_blocks = convertToTextBlocks(extracted_text)

        organised_block = {}

        print("Organising text into JSON blocks")
        # Add tqdm for the outer loop over divisions
        for division, families in text_blocks.items():

            # Add tqdm for the inner loop over families
            for family in tqdm(families, desc=f"Processing Families in {division}", unit="family", leave=True):
                json_conversation = self.prompt.get_conversation(family)
                json_text = self.model(json_conversation, None, debug)
                if division in organised_block:
                    organised_block[division].extend(json_text)
                else:
                    organised_block[division] = json_text

        return organised_block

