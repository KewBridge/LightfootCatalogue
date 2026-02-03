# Python Modules
import os
from tqdm import tqdm
from typing import Optional, Union

# Import Custom Modules
import lightcat.config as config
from lightcat.models import get_model
from lightcat.models.base_model import BaseModel
from lightcat.utils.prompt_utils import PromptLoader
from lightcat.utils.file_utils import save_to_file, get_save_file_name
from lightcat.utils.save_utils import save_json, save_csv_from_json, verify_json
from lightcat.data_processing.text_processing import TextProcessor
# Logging
from lightcat.utils import get_logger
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
        
        super().__init__(prompt)
        self.temperature = self.prompt.get("transcription_temperature", 0.1)    
        self.text_processor = TextProcessor()
        self.model = None
        self.transcription_batch_size = config.TRANSCRIPTION_BATCH_SIZE
        self.save_interval = config.SAVE_JSON_INTERVAL
        
        # Statistics tracking
        self.stats = {
            "total_blocks": 0,
            "successful_blocks": 0,
            "failed_blocks": 0,
            "corrected_blocks": 0
        }


    def _prepare_batch_conversation(self, text_blocks: list[dict]) -> tuple[list, list]:
        """
        Prepare batch conversations for model inference.

        Args:
            text_blocks: List of text blocks to process
        
        Returns:
            A tuple containing:
                - List of conversations for the model
                - List of corresponding division-family identifiers
        """

        batch_conversations = []
        batch_metadata = []

        for text_block in text_blocks:
            division = text_block["division"]
            family = text_block.get("family", division) or ""
            content = text_block["content"]

            conversation = self.promptBuilder.get_conversation(family + "\n" + content)
            batch_conversations.append(conversation)
            batch_metadata.append(dict(
                division=division,
                family=family,
                content=content
            ))

        return batch_conversations, batch_metadata

    
    def batch_inference(self, 
                        batch_conversations: list,
                        debug: bool) -> list:
        """
        
        Perform batch inference. In the case of error handling, process each conversation individually.

        Args:
            batch_conversations (list): List of conversations for the model

        Returns:
            list: List of model outputs
        """

        try:
            batch_outputs = self.model(batch_conversations, None, debug)
            return batch_outputs
        except Exception as e:
            logger.error(f"Batch inference failed: {e}. Processing individually.")
            batch_outputs = []
            for conversation in batch_conversations:
                try:
                    output = self.model(conversation, None, debug)
                except Exception as e2:
                    logger.error(f"Individual inference failed: {e2}. Skipping conversation.")
                    output = None
                batch_outputs.append(output)
            return batch_outputs


    def verify_json_integrity(self, json_text: str, family: str, debug: bool) -> tuple[bool, Optional[dict]]:
        """
        Verify integrity of output

        Args:
            json_text (str): The JSON text output from the model
            family (str): The family name for logging
            debug (bool): Enable debug logging

        Returns:
            tuple[bool, Optional[dict]]: A tuple containing:
                - A boolean indicating if the JSON is verified
                - The loaded JSON dictionary if verified, else None
        """

        json_verified, json_loaded = verify_json(
            json_text, 
            clean=True, 
            out=True, 
            schema=self.prompt.get_schema()
            )
        
        if json_verified:
            return True, json_loaded

        preview = json_text if len(json_text) <= 1200 else json_text[:1200] + "..."
        logger.warning(f"Raw model output for {family} (preview): {preview}")

        logger.warning(f"JSON validation failed for {family}. Attempting correction...")

        error_fix_prompt = self.promptBuilder.getJsonPrompt(json_text)
    
        try:
            corrected_json = self.model([error_fix_prompt], None, debug)
            json_verified, json_loaded = verify_json(
                corrected_json[0], 
                clean=True, 
                out=True, 
                schema=self.prompt.get_schema()
            )
            
            if json_verified:
                self.stats["corrected_blocks"] += 1
                logger.info(f"JSON correction successful for {family}.")
                return True, json_loaded
            else:
                logger.error(f"JSON correction failed for {family}. Data will be skipped")
                return False, None
        except Exception as e:
            logger.error(f"Error during correction for {family}: {e}")
            return False, None

    def process_inference_result(self, 
                             metadata: dict, 
                             json_text: str, 
                             organised_blocks: dict,
                             error_text_file: str,
                             debug: bool) -> bool:
        """
        Process a single inference result, verify JSON, and add to organised blocks.
        
        Args:
            metadata: Block metadata (division, family, content)
            json_text: Raw JSON output from model
            organised_blocks: Dictionary to accumulate organized blocks
            error_text_file: Path to error log file
            debug: Enable debug logging
            
        Returns:
            True if block was successfully processed, False otherwise
        """
        division = metadata["division"]
        family = metadata["family"]
        content = metadata["content"]
        
        if json_text is None:
            self.stats["failed_blocks"] += 1
            save_to_file(error_text_file, f"{division} | {family} | {content}\n", mode="a")
            return False
        
        # Verify and correct JSON
        json_verified, json_loaded = self.verify_json_integrity(json_text, family, debug)
        
        if not json_verified:
            self.stats["failed_blocks"] += 1
            save_to_file(error_text_file, f"{division} | {family} | {content}\n", mode="a")
            return False
        
        # Add to organized blocks
        self.stats["successful_blocks"] += 1
        if division in organised_blocks:
            organised_blocks[division].append(json_loaded)
        else:
            organised_blocks[division] = [json_loaded]
        
        return True
    

    def inference(self, 
                  text_blocks: list[dict], 
                  save_file_name: str, 
                  json_file_name: Optional[str] = None, 
                  save: bool = False, 
                  debug: bool = False) -> dict:
        """
        Perform inference on text blocks to generate structured JSON.
        
        Args:
            text_blocks: Dictionary of text blocks to process
            save_file_name: Base name for output files
            json_file_name: Specific JSON filename (optional)
            save: Whether to save incrementally
            debug: Enable debug logging
            
        Returns:
            Dictionary of organized blocks by division
        """
        logger.info("Organising text into JSON blocks")
        json_file_name = save_file_name + ".json" if json_file_name is None else json_file_name
        error_text_file = os.path.join(self.save_path, save_file_name + "_errors.txt")
        organised_blocks = {}
        # Add tqdm for the outer loop over division
        save_counter = 0

        # Reset statistics
        self.stats = {
            "total_blocks": 0,
            "successful_blocks": 0,
            "failed_blocks": 0,
            "corrected_blocks": 0
        }

        total_blocks = len(text_blocks)
        logger.info(f"Processing a total of {total_blocks} text blocks with batch size {self.transcription_batch_size}")

        bar = tqdm(
            range(0, total_blocks, self.transcription_batch_size),
            desc="Processing text blocks",
            unit="batch",
            leave=True
        )
        

        for batch_start_idx in bar:

            # Get the batch of text blocks
            batch_end_idx = min(batch_start_idx + self.transcription_batch_size, total_blocks)
            batch_text = text_blocks[batch_start_idx: batch_end_idx]

            # Process the batch of text blocks into the LLM conversation and metadata
            batch_conversations, batch_metadata = self._prepare_batch_conversation(batch_text)

            json_outputs = self.batch_inference(batch_conversations, debug)

            for metadata, json_text in zip(batch_metadata, json_outputs):
                self.stats["total_blocks"] += 1
                successfully_parsed = self.process_inference_result(
                    metadata, 
                    json_text, 
                    organised_blocks, 
                    error_text_file, 
                    debug
                )

                if successfully_parsed:
                    save_counter += 1
                    # save to file after n iterations
                    if save and (save_counter >= self.save_interval):
                        save_counter = 0
                        save_json(organised_blocks, json_file_name, self.save_path)
                        save_csv_from_json(os.path.join(self.save_path, json_file_name), save_file_name, self.save_path)
        
        # Log final statistics
        logger.info(f"Transcription Statistics:")
        logger.info(f"  Total blocks: {self.stats['total_blocks']}")
        logger.info(f"  Successful: {self.stats['successful_blocks']}")
        logger.info(f"  Corrected: {self.stats['corrected_blocks']}")
        logger.info(f"  Failed: {self.stats['failed_blocks']}")
        if self.stats['failed_blocks'] > 0:
            logger.warning(f"  Success rate: {(self.stats['successful_blocks']/self.stats['total_blocks']*100):.1f}%")
    
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
        # Use configured batch size instead of hardcoded value
        
        # Converting the extracted text into text blocks defined by divisions and families
        logger.info("Converting extracted text into Text Blocks")
        text_structure = self.text_processor(extracted_text, divisions=self.prompt.get_divisions(), max_chunk_size=max_chunk_size)

        text_blocks = self.text_processor.make_text_blocks(text_structure)
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
