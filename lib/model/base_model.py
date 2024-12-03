import os
from tqdm import tqdm
import time
import json
import pandas as pd
from lib.model import get_model
from lib.utils.promptLoader import PromptLoader
from lib.utils.utils import debugPrint
from lib.utils.json_utils import save_jsons, verify_json

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
                 timeout: int = 4,
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
        self.save_path = save_path if save_path is not None else "./"
        self.timeout = timeout
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
        """
        (Private function)
        Saves the extracted text to the temporary file

        """

        with open(self.TEMP_TEXT_FILE, "w") as f:
            f.write(text)
    

    def _load_from_temp(self):
        """
        (Private function)
        Load the extracted text from the temporary file

        Returns:
            text: the text read from the file

        """

        text = ""

        with open(self.TEMP_TEXT_FILE, "r") as f:
            text = f.read()
        
        return text
    

    def extract_text(self, images: list, debug: bool = False):
        """
        Iterate through all images and extract the text from the image, saving at intervals.
        Combine all extracted text into one long text

        Parameters:
            images: a list of all images to extract from
            debug: used when debugging. Prints debugPrint() messages
        
        Returns:
            joined_text: a combined form of all the text extracted from the images.
        """
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

    def save_final_out(self, orgnasied_blocks: dict, file_name: str = "sample"):
        """
        Saves final output of model as JSON and CSV file at given location

        Parameters:
            organised_blocks (str): the json dict output from the model
            file_name (str): the file name to save as
        """

        
        file_path = os.path.join(self.save_path, file_name +".json")
        csv_file_path = os.path.join(self.save_path, file_name + ".csv")
        # Saving JSON file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(orgnasied_blocks, json_file, indent=4)

        # Saving CSV file

        with open(file_path, "r", encoding="utf-8") as json_file:
            pandas_json = pd.read_json(json_file)
        
        pandas_json.to_csv(csv_file_path, encoding="utf-8", index=False)



    def __call__(self, images: list, text_file: str = None, save: bool = False, save_file_name: str = "sample", debug: bool = False) -> list:
        """
        The main pipeline that extracts text from the images, seperates them into text blocks and organises them into JSON objects

        Paramaters:
            images (list): a list of images to extract text from
            text_file (str): the path to the text file containing the pre-extracted text to use
            save (bool): Boolean to determine whether to save the outputs or not
            save_file_name (str): the name of the save files
            debug (bool): used when debugging. Prints debugPrint() messages
        """

        self.info()

        #===================================================
        # Extracting text from image or loading a temp file
        #===================================================
        if text_file is None:
            print("Extracting Text from Images")
            extracted_text = self.extract_text(images, debug)
        else:
            print("Skipping extraction...")
            print("Loading text from provided extracted text file")
            with open(text_file, "r") as file_:
                extracted_text = file_.read()

        #===================================================
        # Converting the extracted text into text blocks defined by divisions and families
        #===================================================
        print("Converting extracted text into Text Blocks")
        text_blocks = convertToTextBlocks(extracted_text)


        #===================================================
        # Performing inference on the text blocks to generate JSON files
        #===================================================
        organised_blocks = {}

        print("Organising text into JSON blocks")
        # Add tqdm for the outer loop over divisions
        for division, families in text_blocks.items():

            # Add tqdm for the inner loop over families
            for family in tqdm(families, desc=f"Processing Families in {division}", unit="family", leave=True):
                # Load the system conversation with text blocks added to the prompt
                json_conversation = self.prompt.get_conversation(family)

                # Start a while loop for a count of timeout
                count = 0
                while count < self.timeout:
                    # Perform inference on text
                    json_text = self.model(json_conversation, None, debug)
                    # Check the integrity of the JSON output. 
                    # Json verified is boolean to check if the integrity of the JSON output is valid
                    # Json loaded is the post-processed form of the text into dict (removing and cleaning done)
                    json_verified, json_loaded = verify_json(json_text, clean=True, out=True)
                    
                    # If verified, add to organised block and break
                    if json_verified:
                        if division in organised_blocks:
                            organised_blocks[division].extend(json_loaded)
                        else:
                            organised_blocks[division] = json_loaded
                        break

                    count += 1
                    time.sleep(1) # Adding a delay to not overwhelm the system

        #===================================================
        # Saving the outputs if prompted
        #===================================================        
        if save:
            self.save_final_out(organised_blocks, save_file_name)
        
        return organised_blocks

