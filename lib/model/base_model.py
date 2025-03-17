# Python Modules
import os
from tqdm import tqdm
import logging
from typing import Optional

# Import Custom Modules
from lib.model import get_model
from lib.utils.promptLoader import PromptLoader
from lib.utils.save_utils import save_json, save_csv_from_json, verify_json
from lib.utils.text_utils import convertToTextBlocks
from lib.utils.text_processing import TextProcessor
logger = logging.getLogger(__name__)

class BaseModel:

    TEMP_TEXT_FILE = "temp.txt"
    SAVE_TEXT_INTERVAL = 3

    def __init__(self, 
                 model_name: str,
                 prompt: Optional[str] = None,
                 batch_size: int = 3, 
                 max_new_tokens: int = 5000,
                 temperature: float = 0.2, 
                 save_path: Optional[str]= None, 
                 timeout: int = 4,
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
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.save_path = save_path if save_path is not None else "./outputs/"
        
        if not(os.path.isdir(self.save_path)):
            os.makedirs(self.save_path)

        # Defining a temperory file to store extracted text
        self.TEMP_TEXT_FILE = os.path.join(self.save_path, self.TEMP_TEXT_FILE)

        self.timeout = timeout
        
        # Load the model, prompt, and the conversation
        self.model = get_model(self.model_name)(self.batch_size, self.max_new_tokens, self.temperature, **kwargs)
        self.prompt = PromptLoader(prompt)#prompt
        self.text_processor = TextProcessor()

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


    def extract_text(self, images: list[str], save_file: str = None, debug: bool = False) -> str:
        """
        Iterate through all images and extract the text from the image, saving at intervals.
        Combine all extracted text into one long text

        Parameters:
            images (list): a list of all images to extract from
            save_file (str): Path to save file
            debug (bool): used when debugging. logs debug messages
        
        Returns:
            joined_text (str): a combined form of all the text extracted from the images.
        """
        if debug:
            logger.debug("Batching Images...")


        batch_texts = []
        # Create batches of images
        batched_images = [images[x:min(x + self.batch_size, len(images))] for x in range(0, len(images), self.batch_size)]

        # Add tqdm for progress tracking
        for ind, batch in enumerate(tqdm(batched_images, desc="Processing Batches", unit="batch")):
            #print(f">>> Batch {ind + 1} starting...")

            if debug:
                logger.debug("Extracting text from image")
            image_conversation = self.prompt.getImagePrompt()
            extracted_text = self.model(image_conversation, batch, debug)
            
            if debug:
                logger.debug("\tJoining Outputs...")
            # Join all the text and append it together with previous batches
            batch_texts.append("\n\n".join(extracted_text))

            if (ind + 1) % self.SAVE_TEXT_INTERVAL == 0:
                if debug:
                    logger.debug("\tStoring at interval...")
                self._save_to_file(self.TEMP_TEXT_FILE if save_file is None else save_file, "\n\n".join(batch_texts))

            if debug:
                logger.debug("\tBatch Finished")

        return "\n\n".join(batch_texts)
    

    def get_extracted_text(self, images: list[str], text_file: Optional[str] = None, save_file: str = None, debug: bool = False) -> str:
        """
        Extracting text from image or loading a temp file

        Paramaters:
            images (list): a list of images to extract text from
            text_file (str): the path to the text file containing the pre-extracted text to use
            save_file (str): Path to save file
            debug (bool): used when debugging. logs debug messages

        Returns:
            extracted_text (str): Extracted text as a long string
        """
        
        if text_file is None:
            logger.info("Extracting Text from Images")
            extracted_text = self.extract_text(images, save_file, debug)
        else:
            logger.info("Skipping extraction...")
            logger.info(f"Loading text from provided extracted text file `{text_file}`")
            with open(text_file, "r") as file_:
                extracted_text = file_.read()
        
        return extracted_text

    def inference(self, 
                  text_blocks: dict, 
                  save_file_name: str, 
                  json_file_name: Optional[str] = None, 
                  save: bool = False, 
                  debug: bool = False) -> dict:

        
        json_file_name = save_file_name + ".json" if json_file_name is None else json_file_name
        error_text_file = os.path.join(self.save_path, save_file_name + "_errors.txt")
        organised_blocks = {}

        logging.info("Organising text into JSON blocks")
        # Add tqdm for the outer loop over division
        save_counter = 0
        for item in tqdm(text_blocks, desc="Processing text blocks", leave=True):
            division = item["division"]
            family = item["family"] if "family" in item.keys() else division
            content = item["content"]
            # Add tqdm for the inner loop over families
            self._save_to_file(error_text_file, f"{division}\n", mode="a")
            # Load the system conversation with text blocks added to the prompt
            json_conversation = self.prompt.get_conversation(family+"\n"+content)

            # Perform inference on text
            json_text = self.model(json_conversation, None, debug)
            
            # Check the integrity of the JSON output. 
            # Json verified is boolean to check if the integrity of the JSON output is valid
            # Json loaded is the post-processed form of the text into dict (removing and cleaning done)
            json_verified, json_loaded = verify_json(json_text[0], clean=True, out=True)
            
            if not(json_verified):
                logging.info("Error Noticed in JSON")
                logging.info("Fixing Error")
                error_fix_prompt = self.prompt.getJsonPrompt(json_text[0])
                # print(error_fix_prompt)
                json_text = self.model(error_fix_prompt, None, debug)
                # print(json_text)
                json_verified, json_loaded = verify_json(json_text[0], clean=True, out=True)

                # storing all erroneous JSON format in error.txt
                self._save_to_file(error_text_file, f"{family}\n", mode="a")
                    
                
            # If verified, add to organised block and break
            if json_verified:
                if division in organised_blocks:
                    organised_blocks[division].append(json_loaded)
                else:
                    organised_blocks[division] = [json_loaded]
                                    
            save_counter += 1
            # save to file after 10 iterations
            if save and (save_counter == 10):
                save_counter = 0
                save_json(organised_blocks, json_file_name, self.save_path)
                save_csv_from_json(os.path.join(self.save_path, json_file_name), save_file_name, self.save_path)
    
        return organised_blocks

    def __call__(self,
                 extracted_text: Optional[str] = None,
                 images: Optional[list[str]] = None,
                 save: bool = False,
                 save_file_name: str = "sample",
                 max_chunk_size: int = 3000,
                 debug: bool = False) -> dict:
        """
        The main pipeline that extracts text from the images, seperates them into text blocks and organises them into JSON objects

        Paramaters:
            extracted_text (str): The extracted text from the images
            images (list): a list of images to extract text from
            save (bool): Boolean to determine whether to save the outputs or not
            save_file_name (str): the name of the save files
            debug (bool): used when debugging. logs debug messages

        Returns:
            organised_blocks (dict): Extracted data organised in a JSON format
        """

        self.info()
        save_file_name = self._get_save_file_name(save_file_name)
        json_file_name = save_file_name + ".json"
        
        logging.info(f"""Saving data into following files at {self.save_path}: \n
                     \t==> JSON file: {save_file_name}.json\n
                     \t==> CSV file: {save_file_name}.csv
                     \t==> Errors: {save_file_name}_errors.txt
                     """)
        # Get the extracted text whether from file or from images
        if extracted_text is None or extracted_text == "":
            extracted_text = self.get_extracted_text(images, None, debug)
        
        
        # Converting the extracted text into text blocks defined by divisions and families
        logging.info("Converting extracted text into Text Blocks")
        text_structure = self.text_processor(extracted_text, divisions=self.prompt.get_divisions(), max_chunk_size=max_chunk_size)
        text_blocks = self.text_processor.make_text_blocks(text_structure)

        # Performing inference on the text blocks to generate JSON files
        organised_blocks = self.inference(text_blocks, save_file_name, json_file_name, save, debug)
        
        # Saving the outputs if prompted       
        if save:
            save_json(organised_blocks, json_file_name, self.save_path)
            save_csv_from_json(os.path.join(self.save_path, json_file_name), save_file_name, self.save_path)
        
        return organised_blocks
