# Python Modules
import os
import unicodedata
from tqdm import tqdm
from typing import Optional, Union
import re
import cv2
from pytesseract import image_to_string
import numpy as np
import gc
from PIL import Image

# Import Custom Modules
from lightcat.utils.file_utils import save_to_file, load_from_file, get_save_file_name
from lightcat.models.base_model import BaseModel
from lightcat.utils.prompt_utils import PromptLoader
from lightcat.utils.save_utils import save_json, save_csv_from_json, verify_json
from lightcat.data_processing.text_processing import TextProcessor
from lightcat.data_processing.chunker import SpeciesChunker
from lightcat.data_processing.layout_detection import LayoutDetector
import lightcat.config as config
# Logging
from lightcat.utils import get_logger
logger = get_logger(__name__)

class OCRModel(BaseModel):
    """OCR Model for extracting and cleaning text from images."""
    
    def __init__(self, prompt: Union[Optional[str], PromptLoader] = None):

        """OCR model for text extraction from images.

        Parameters:
            prompt: The name of the prompt file, path to it, or PromptLoader instance
        """
        super().__init__(prompt)
        self.temperature = self.prompt.get("ocr_temperature", 0.1)
        self.model_name = self.prompt.get("ocr_model", "qwen2.5")
        self.extraction_model_name = self.prompt.get("ocr_extraction_model", "qwen2.5")
        self.save_interval = config.SAVE_TEXT_INTERVAL
        self.model = self.load_model()
        self.extraction_model = None
        self.chunker = SpeciesChunker()
        self.layout_detection = None


    def load_ld(self):
        self.layout_detection = LayoutDetector()

    def getImagePrompt(self) -> list:
        """
        Get the image extraction prompt

        Returns:
            list: Image extraction conversation to VL model
        """

        system_prompt = (
            "You are an expert in extracting verbatim text from images."
        )

        image_prompt = (
            "Please perform OCR on this image. Return all verbatim text without any explanation." 
        )

        return self.promptBuilder.getImagePrompt(system_prompt, image_prompt)
    

    def getOCRNoiseCleaningPrompt(self, extracted_text: str, context: str="") -> list:
        """
        Get OCR noise cleaning prompt
        This is used to clean the extracted text from the images.

        Parameters:
            extracted_text (str): Extracted text from the images

        Returns:
            list: ocr noise cleaning conversation to the model
        """

       

        system_prompt = (
            "You are an expert in cleaning OCR text\n"
            "You will be provided with a text containing botanical information from a historical botanical catalogue.\n"
            "The text contains botanical information, including family names, species names, and other relevant details.\n"
            "This information denotes the how each speciemen is stored in the catalogue.\n"
            "Your task is to clean the text by following the rules:\n"
            "1. Find and clean any OCR artefacts, like missing spaces, incorrect characters, or formatting issues.\n"
            "2. Join any words that are split across lines, ensuring that the meaning is preserved. Ensure the lines joined are contextually appropriate.\n"
            "3. Only return the cleaned text, without any additional comments or explanations.\n"

        )

        user_prompt = (
            "By following the rules cleaned the following OCR'd text:\n\n"
            f"{extracted_text}\n"
        )

        return self.promptBuilder.getTextPrompt(system_prompt, user_prompt)

    def clean(self, text: str) -> str:
        """
        Clean any headings added by the model and remove any unwanted text

        Parameters:
            text (str): text in need of cleaning

        Returns:
            str: Cleaned text
        """

        text = re.sub(r"^(Cleaned|Corrected)\stext\s{0,1}:\s*", "", text, flags=re.IGNORECASE)

        return text
    
    def post_process(self, text: str) -> str:

        return text
    

    def single_image_extract(self, image: Union[str, np.ndarray]) -> str:

        logger.debug("Extracting text from image...")

        if isinstance(image, np.ndarray):

            image = Image.fromarray(image)

        text = ""
        if (self.extraction_model_name is None) or (self.extraction_model_name.lower() == "default"):
            text = image_to_string(image, lang="eng+lat", config="--psm 1")
        else:
            print(f"Using {self.extraction_model_name} for text extraction.")
            if self.extraction_model is None:
                self.extraction_model = self.load_model(self.extraction_model_name)
            message = self.getImagePrompt()
            text = self.extraction_model(message, [image])[0]
        
        return text.strip()
    
    

    def clean_text(self, text: str) -> str:
        """
        Clean input text for any OCR noise

        Args:
            text (str): noisy text

        Returns:
            str: cleaned text
        """

        # Unicode normalization
        text = unicodedata.normalize("NFKD", text)

        replacements = {
            "‘" : "'",
            "’" : "'",
            "“" : '"',
            "”" : '"',
            "–" : "-",
            "—" : "-",
            "…" : "...",
        }

        # Creating a transalation table for replacements of characters
        trans_table = str.maketrans(replacements)
        text = text.translate(trans_table)

        # Remove page numbers
        text = re.sub(r"(?m)^\s*\d+\s*$\n?", r"", text)
        # Fix hyphenated line breaks
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        # Fix Family name variants
        text = re.sub(r"([A-Z]+)\s*(EAE|FAE|EAF)", r"\1EAE", text)
        # Collapse multiple newlines
        text = re.sub(r"\n{2,}", r"\n", text) 
        text = re.sub(r"[ \t]{2,}", " ", text).strip()

        # Ensure family names are on their own lines
        legacy_alts = "|".join([re.escape(alt) for alt in config.LEGACY_FAMILY_NAMES])

        family_regex = rf"\b([A-Z]+ACEAE|{legacy_alts})\b"
        

        text = re.sub(family_regex, r"\n\n\1\n\n", text)  # Find all uppercase words
        text = re.sub(r"\n{3,}", r"\n\n", text)  # Collapse multiple newlines

        return text.strip()
    

    def extract_text(self, 
                     images: list[str], 
                     save_file: str = None, 
                     debug: bool = False, 
                     clean: bool = True) -> str:
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

        batch_texts = []
        # Create batches of images
        save_file_name = self.TEMP_TEXT_FILE if save_file is None else save_file
        #Add previous block of text to the next batch
        # This is done to ensure that the model does not forget the previous text
        # Add tqdm for progress tracking
        for ind, image in enumerate(tqdm(images, desc="Processing images", unit="image")):
            extracted_text = self.single_image_extract(image)
            cleaned_text = self.clean_text(extracted_text) if clean else extracted_text
            batch_texts.append("\n" + cleaned_text)

            if (ind + 1) % self.save_interval == 0:
                save_to_file(save_file_name, "\n\n".join(batch_texts), mode="a")
                batch_texts = []
                # Freeing up memory after checkpoint
                gc.collect() 
        
        if batch_texts:
            save_to_file(save_file_name, "\n\n".join(batch_texts), mode="a")
        

        return load_from_file(save_file_name)
    

    def chunk_and_clean(self, text: str, add_overlap: bool = True) -> list[str]:

        """
        Chunk the text into smaller chunks for cleaning and processing.
        This is done to ensure that the model does not run out of memory when processing large texts.

        Parameters:
            text (str): The text to chunk and clean
            add_overlap (bool): Whether to add overlap between chunks

        Returns:
            list: A list of cleaned chunks
        """
        logger.info("Chunking text for cleaning...")
        chunks = self.chunker.chunk_text_for_cleaning(text, add_overlap=add_overlap)

        logger.info("Cleaning chunks...")
        cleaned_chunks = []
        context = ""
        for chunk in tqdm(chunks, desc="Cleaning Chunks", unit="chunk"):
            cleaned_chunks.append(chunk)

        merged_text = self.chunker.merge_sentences(cleaned_chunks)
        return merged_text


    def __call__(self, images: str, text_file: Optional[str] = None, save_file: str = None, debug: bool = False) -> str:
        """
        Extracting text from image or loading a temp file

        Paramaters:
            images (str): the path to a directory of images or a path to a single image
            text_file (str): the path to the text file containing the pre-extracted text to use
            save_file (str): Path to save file
            debug (bool): used when debugging. logs debug messages

        Returns:
            extracted_text (str): Extracted text as a long string
        """

        
        self.info()

        
        if text_file is None:

            logger.info(f"Processing input images before extraction...")
            self.load_ld()
            processed_images = self.layout_detection(images)
            logger.info(f"Extracting text from images...")
            extracted_text = self.extract_text(processed_images, save_file, debug)
            #logger.info(f"Chunking text for cleaning...")
            #extracted_text = self.chunk_and_clean(extracted_text, add_overlap=True)

            del self.extraction_model
            gc.collect()
            # Overwrite the existing text file with the cleaned text
            save_to_file(self.TEMP_TEXT_FILE if save_file is None else save_file, extracted_text)
        else:
            logger.info("Skipping extraction...")
            logger.info(f"Loading text from provided extracted text file `{text_file}`")
            with open(text_file, "r") as file_:
                extracted_text = file_.read()
        
        return extracted_text
