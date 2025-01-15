import os
from tqdm import tqdm
import time
import json
import pandas as pd
from lib.model import get_model
from lib.utils.promptLoader import PromptLoader
from lib.utils.utils import debugPrint
from lib.utils.json_utils import verify_json
from lib.utils.save_utils import save_json, save_csv_from_json
from lib.utils.text_utils import convertToTextBlocks


class BaseModel:

    TEMP_TEXT_FILE = "temp.txt"
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
        self.save_path = save_path if save_path is not None else "./outputs/"
        
        if not(os.path.isdir(self.save_path)):
            os.makedirs(self.save_path)

        self.TEMP_TEXT_FILE = os.path.join(self.save_path, self.TEMP_TEXT_FILE)

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

    def _save_to_file(self, file, text, mode="w"):
        """
        (Private function)
        Saves the extracted text to the file

        """

        with open(file, mode) as f:
            f.write(text)
    

    def _load_from_file(self, file):
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


    def _get_save_file_name(self, save_file_name):

        files = [file for file in os.listdir(self.save_path) if os.path.isfile(file)]

        id = 0

        while True:

            if id == 0:
                final_save_file_name = save_file_name
            else:
                final_save_file_name = save_file_name +  f"_{id}"

            if not((final_save_file_name + ".json") in files):
                return final_save_file_name
            
            id += 1


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
                self._save_to_file(self.TEMP_TEXT_FILE, joined_text)

            debugPrint("\tBatch Finished", debug)

        return joined_text

    def __call__(self, images: list, text_file: str = None, save: bool = False, save_file_name: str = "sample", max_chunk_size: int = 3000, debug: bool = False) -> list:
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
        save_file_name = self._get_save_file_name(save_file_name)
        json_file_name = save_file_name + ".json"
        error_text_file = os.path.join(self.save_path, save_file_name + "_errors.txt")
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
        text_blocks = convertToTextBlocks(extracted_text, divisions=self.prompt.get_divisions(), max_chunk_size=max_chunk_size)


        #===================================================
        # Performing inference on the text blocks to generate JSON files
        #===================================================
        organised_blocks = {}

        print("Organising text into JSON blocks")
        # Add tqdm for the outer loop over divisions
        for division, families in text_blocks.items():
            save_counter = 0
            # Add tqdm for the inner loop over families
            self._save_to_file(error_text_file, f"{division}\n", mode="a")
            for family in tqdm(families, desc=f"Processing Families in {division}", unit="family", leave=True):
                # Load the system conversation with text blocks added to the prompt
                json_conversation = self.prompt.get_conversation(family)

                # Perform inference on text
                json_text = self.model(json_conversation, None, debug)
                
                # Check the integrity of the JSON output. 
                # Json verified is boolean to check if the integrity of the JSON output is valid
                # Json loaded is the post-processed form of the text into dict (removing and cleaning done)
                json_verified, json_loaded = verify_json(json_text[0], clean=True, out=True)
                # Start a while loop for a count of timeout
                #count = 0
                #while count < self.timeout:
                    #print("="*10)
                    # If not verified
                    # TODO: Rework the JSON error fixing code
                if not(json_verified):
                    # print("Error Noticed in JSON")
                    # print("Fixing Error")
                    # error_fix_prompt = self.prompt.getJsonPrompt(json_text[0], json_loaded)
                    # print(error_fix_prompt)
                    # json_text = self.model(error_fix_prompt, None, debug)
                    # print(json_text)
                    # json_verified, json_loaded = verify_json(json_text[0], clean=True, out=True)

                    # storing all erroneous JSON format in error.txt
                    self._save_to_file(error_text_file, f"{family}\n", mode="a")
                        
                    
                # If verified, add to organised block and break
                if json_verified:
                    if division in organised_blocks:
                        organised_blocks[division].append(json_loaded)
                    else:
                        organised_blocks[division] = [json_loaded]
                    #break
                                        
                #count += 1
                #time.sleep(1) # Adding a delay to not overwhelm the system
                print("=" * 10)
                save_counter += 1
                # save to file after 10 iterations
                if save and (save_counter == 10):
                    save_counter = 0
                    save_json(organised_blocks, json_file_name, self.save_path)
                    save_csv_from_json(os.path.join(self.save_path, json_file_name), save_file_name, self.save_path)

        #===================================================
        # Saving the outputs if prompted
        #===================================================        
        if save:
            save_json(organised_blocks, json_file_name, self.save_path)
            save_csv_from_json(os.path.join(self.save_path, json_file_name), save_file_name, self.save_path)
        
        return organised_blocks

