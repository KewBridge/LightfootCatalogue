# Python Modules
import os
from tqdm import tqdm
from typing import Optional, Union

# Import Custom Modules
from lib.model import get_model
from lib.model.base_model import BaseModel
from lib.utils.promptLoader import PromptLoader
from lib.utils.save_utils import save_json, save_csv_from_json, verify_json
from lib.data_processing.text_processing import TextProcessor
# Logging
from lib.utils import get_logger
logger = get_logger(__name__)


class TranscriptionModel(BaseModel):


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
        
        super().__init__(prompt, **kwargs)
        self.temperature = prompt["transcription_temperature"] if prompt["transcription_temperature"] is not None else 0.1

        self.text_processor = TextProcessor()
        self.model = self.load_model()

    

    def inference(self, 
                  text_blocks: dict, 
                  save_file_name: str, 
                  json_file_name: Optional[str] = None, 
                  save: bool = False, 
                  debug: bool = False) -> dict:

        
        json_file_name = save_file_name + ".json" if json_file_name is None else json_file_name
        error_text_file = os.path.join(self.save_path, save_file_name + "_errors.txt")
        organised_blocks = {}

        logger.info("Organising text into JSON blocks")
        # Add tqdm for the outer loop over division
        save_counter = 0

        bar = tqdm(
            text_blocks,
            desc="Processing text blocks", 
            leave=True
            )
        
        for item in bar:
            division = item["division"]
            family = item["family"] if "family" in item.keys() else division
            family = family if family else ""
            content = item["content"]
            bar.set_description(f"Processing division: {division} | family: {family}")
            # Add tqdm for the inner loop over families
            self._save_to_file(error_text_file, f"{division}\n", mode="a")
            # Load the system conversation with text blocks added to the prompt
            json_conversation = self.prompt.get_conversation(family+"\n"+content)
            
            # Perform inference on text
            json_text = self.model(json_conversation, None, debug)
            
            # Check the integrity of the JSON output. 
            # Json verified is boolean to check if the integrity of the JSON output is valid
            # Json loaded is the post-processed form of the text into dict (removing and cleaning done)
            json_verified, json_loaded = verify_json(json_text[0], clean=True, out=True, schema=self.prompt.get_schema())
            
            # A second check to verify the JSON output if repair_json does not work
            if not(json_verified):
                logger.info("Error Noticed in JSON")
                logger.info("Fixing Error")
                error_fix_prompt = self.prompt.getJsonPrompt(json_text[0])
                # print(error_fix_prompt)
                json_text = self.model(error_fix_prompt, None, debug)
                # print(json_text)
                json_verified, json_loaded = verify_json(json_text[0], clean=True, out=True, schema=self.prompt.get_schema())

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
        
        logger.info(f"""Saving data into following files at {self.save_path}: \n
                     \t==> JSON file: {save_file_name}.json\n
                     \t==> CSV file: {save_file_name}.csv
                     \t==> Errors: {save_file_name}_errors.txt
                     """)
        # Get the extracted text whether from file or from images
        if extracted_text is None or extracted_text == "":
            raise ValueError("No extracted text provided. Please provide a valid text file or images to extract from.")
        
        
        # Converting the extracted text into text blocks defined by divisions and families
        logger.info("Converting extracted text into Text Blocks")
        text_structure = self.text_processor(extracted_text, divisions=self.prompt.get_divisions(), max_chunk_size=max_chunk_size)
        text_blocks = self.text_processor.make_text_blocks(text_structure)

        # Performing inference on the text blocks to generate JSON files
        organised_blocks = self.inference(text_blocks, save_file_name, json_file_name, save, debug)
        
        # Saving the outputs if prompted       
        if save:
            save_json(organised_blocks, json_file_name, self.save_path)
            save_csv_from_json(os.path.join(self.save_path, json_file_name), save_file_name, self.save_path)
        
        return organised_blocks
