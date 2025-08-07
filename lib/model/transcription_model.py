# Python Modules
import os
from tqdm import tqdm
from typing import Optional, Union

# Import Custom Modules
from lib.model import get_model
from lib.model.base_model import BaseModel
from lib.utils.promptLoader import PromptLoader
from lib.utils.file_utils import save_to_file, get_save_file_name
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
        self.temperature = self.prompt.get("transcription_temperature", 0.1)    
        self.text_processor = TextProcessor()
        self.model = None#self.load_model()
        self.cleaning_model_name = "mistral7b"
        self.cleaning_model = None#self.load_model()("mistral7b")

    

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
            save_to_file(error_text_file, f"{division}\n", mode="a")
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
                save_to_file(error_text_file, f"{family}\n", mode="a")
                    
                
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

    def getOCRNoiseCleaningPrompt(self, extracted_text: str, context: str="") -> list:
        """
        Get OCR noise cleaning prompt
        This is used to clean the extracted text from the images.

        Parameters:
            extracted_text (str): Extracted text from the images

        Returns:
            list: ocr noise cleaning conversation to the model
        """

        # system_prompt = (
        #     "You are an expert in cleaning OCR induced errors in the text. \n"
        #     "Follow the instructions below to clean the text, ensuring the text flows coherently with the previous context:\n"
        #     "1. Fix OCR induced typographical errors, such as incorrect characters, spacing and improper symbols.\n"
        #     "- Use provided context and common sense to identify and correct errors.\n"
        #     "- The letter 'AE' and 'Æ' are often confused with symbols such as '&' and other special symbols.\n"
        #     "- For example, 'l' and '1' or 'o' and '0' are often confused.\n"
        #     "- Ensure that the text is grammatically correct and coherent.\n"
        #     "- Remove any unnecessary line breaks or extra spaces.\n"
        #     "- Identify and correct word splits and line breaks.\n"
        #     "- Only fix clear OCR errors. DO NOT ALTER THE CONTEXT OR MEANING of the text.\n"
        #     "- DO NOT add any generated text, punctuation, or capitalization.\n"
        #     "2. Ensure structure is maintained.\n"  
        #     "- Maintain original structure, including paragraphs and line breaks.\n"
        #     "- Preserve the original content. \n"
        #     "- Keep all importatnt information intact.\n"
        #     "- DO NOT add any new text not present in the text. \n"
        #     "3. Ensure flow and coherence.\n"
        #     "- Ensure the text flows naturally and coherently.\n"
        #     "- Use provided context to ensure the text makes sense.\n"
        #     "- HANDLE text that starts or ends mid-sentence correctly. \n\n"
        #     "4. Return ONLY the cleaned text.\n"
        #     "- Do not add any additional information, explanations, or thoughts.\n"
        #     "- Do not include your thoughts, explanations, or steps.\n"
        #     "- Do not add any new text not present in the text.\n"
        # )
        # noise_prompt = (
        #     # "IMPORTATANT: RETURN ONLY THE CLEANED TEXT. Preserve the orignial structure and content. Do not add anything else. Do not include your thoughts, explantions or steps.\n\n"
        #     f"Previous context:\n {context}\n\n"
        #     f"Text to clean:\n {extracted_text}\n\n"
        #     "Cleaned text:\n"
        # )

        system_prompt = (
            "You are an expert in cleaning OCR text\n"
            "You will be provided with a text containing botanical information from a historical botanical catalogue.\n"
            "The text contains botanical information, including family names, species names, and other relevant details.\n"
            "This information denotes the how each speciemen is stored in the catalogue.\n"
            # "You task is to list all ocr artefacts, grammatical errors, and formatting issues in the text.\n"
            # "You will not make any changes to the text.\n"
            # "Do not make any assumption about the text, if you are not sure about something, keep the original text.\n"
            # "Think step by step and provide a detailed analysis of the text.\n"
            # "Return a rating out of 10 for the overall quality of the text.\n"
            "Your task is to clean the text by following the rules:\n"
            "1. Find and clean any OCR artefacts, like missing spaces, incorrect characters, or formatting issues.\n"
            "2. Join any words that are split across lines, ensuring that the meaning is preserved. Ensure the lines joined are contextually appropriate.\n"
            "3. Only return the cleaned text, without any additional comments or explanations.\n"
            #"4. Compare and return an accuracy rating out of 10 between the original and cleaned text. Higher the rating, the more accurate. The returned rating should be at the end of the cleaned text following the strucutre: RATING: <rating>\n"
        )

        user_prompt = (
            "By following the rules cleaned the following OCR'd text:\n\n"
            f"{extracted_text}\n"
        )

        return self.prompt.getTextPrompt(system_prompt, user_prompt)
    
    def clean_post_ocr_text(self, chunked_et: str) -> str:
        """
        Clean the extracted text from the images using the OCR noise cleaning prompt

        Parameters:
            chunked_et (dict): Extracted text from the images chunked using text processor

        Returns:
            str: Cleaned text
        """

        logger.info("Cleaning OCR noise in the extracted text")
        
        for div, family_list in chunked_et.items():
            for family in family_list:
                all_species = family["species"]
                if not all_species:
                    continue

                batch_size = 10
                batches = [all_species[i:i + batch_size] for i in range(0, len(all_species), batch_size)]
                outputs = []
                for batch in batches:
                    # Create a conversation for the batch
                    cleaning_convs = [self.getOCRNoiseCleaningPrompt(i) for i in batch]
                    # Perform inference on the batch
                    output = self.cleaning_model(conversation=cleaning_convs)

                    for out in output:
                        outputs.append(out.strip())
                
                family["species"] = outputs
        logger.info("OCR noise cleaning completed")

        return chunked_et

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
        save_file_name = get_save_file_name(self.save_path, save_file_name)
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

        logger.info("Performing text cleaning on chunked ocr'd text")
        logger.info("Loading the cleaning model")
        self.cleaning_model = self.load_model(self.cleaning_model_name)
        self.cleaning_model.load()
        text_structure = self.clean_post_ocr_text(text_structure)
        text_blocks = self.text_processor.make_text_blocks(text_structure)
        logger.info("Unloading the cleaning model")
        self.cleaning_model.unload()
        logger.info("Text blocks created successfully")

        logger.info("Trasncribing text blocks into JSON format")
        logger.info("Loading the transcription model")
        self.model = self.load_model()
        self.model.load()
        # Performing inference on the text blocks to generate JSON files
        organised_blocks = self.inference(text_blocks, save_file_name, json_file_name, save, debug)
        logger.info("Unloading the transcription model")
        self.model.unload()
        # Saving the outputs if prompted       
        if save:
            save_json(organised_blocks, json_file_name, self.save_path)
            save_csv_from_json(os.path.join(self.save_path, json_file_name), save_file_name, self.save_path)
        
        return organised_blocks
