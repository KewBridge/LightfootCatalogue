# Python Modules
import os
import time
from typing import Optional, Union
from pdf2image import convert_from_path
from natsort import natsorted
from tqdm import tqdm
# Custom Modules
from lib.utils.promptLoader import PromptLoader
import lib.config as config
from lib.data_processing.image_processor import ImageProcessor
from lib.model.ocr_model import OCRModel
# Logging
from lib.utils import get_logger
logger = get_logger(__name__)


class DataReader:

    EXTRACTED_TEXT="extracted_text.txt"
    CROPPED_DIR_NAME="cropped"
    ALLOWED_EXT=["jpeg", "png", "jpg", "pdf"]

    def __init__(self,
                 data_path: str,
                 prompt: PromptLoader
                 ):
        """
          Data reader class used to read and extract text from any source of input (image or pdfs)

          The class aims to read the data, extract the text from the images, save them to a text file.
          In case a text file with the extracted text already exists, the extracted data is loaded (without any need to do another inference)

          Parameters:
              data_path (str): the path to the folder containig the data or the the path to the pdf
              prompt (PromptLoader): the prompt containing all the parameters for extraction
        """
        self.data_path = data_path
        self.extraction_model = OCRModel(prompt=prompt)
        self.prompt = PromptLoader(prompt) if isinstance(prompt, str) else prompt
        self.crop = prompt["crop"] if prompt["crop"] is not None else False

        self.image_processor = ImageProcessor(self.prompt["padding"],prompt["resize_factor"], prompt["remove_area_perc"], prompt["middle_margin_perc"], prompt["double_pages"])       


    def load_files(self, path: Optional[str]=None) -> list[str]:
        """
        Load (Unravel) files (image or pdfs) given a path to a directory or single image

        If a nested directory is given, all images inside said nexted directories are also gathered.

        Parameter:
            path (str): the absolute path to a directory of images or an image

        Return:
            all_files (list): A list of all the possible images/pdfs in a directory
            
        """

        path = self.data_path if path is None else path
        
        extraction_path = os.path.join(path, "extracted_images")
        needs_conversion = (
            not(os.path.isdir(extraction_path)) or len(os.listdir(extraction_path)) == 0
                
        )
        
        # Return just file if path is to an image or pdf (confirms by checking approved extensions)
        if os.path.isfile(path) and (path.split(".")[-1].lower() in self.ALLOWED_EXT):
            if path.split(".")[-1].lower() == "pdf" and needs_conversion:
                logger.info("Detected PDF! Converting to images ...")
                image_paths = self.pdf_to_images(os.path.dirname(path), path)
                return image_paths
            return [self.data_path]
        
        all_files = []

        # Traverse through the directory
        for file in os.listdir(path):
            #print(f"Processing file: {file}")
            if file in config.IGNORE_FILE:
                continue
            # Get the file path
            file_path = os.path.join(path, file)

            # Check if it is an image or pdf
            if not(os.path.isdir(file_path)):
                extension = file.split(".")[-1]
                if (extension in self.ALLOWED_EXT):
                    if extension == "pdf" and needs_conversion:
                        logger.info("Detected PDF! Converting to images ...")
                        image_paths = self.pdf_to_images(path, file_path)

                        all_files.extend(image_paths)
                    else:
                        all_files.append(file_path)
        
        return all_files


    def pdf_to_images(self, main_path: str, pdf_path: str) -> list[str]:
        """
        Convert the input pdf into a set of images stored in a folder

        Parameters:
            main_path (str): folder in which the pdf was found
            pdf_path (str): path to pdf
        
        Returns:
            list[str] -> A list of all path to the new images
        """
        output_dir = os.path.join(main_path, "extracted_images")

        if not(os.path.isdir(output_dir)):
            os.makedirs(output_dir)

        pdf_name = pdf_path.split(os.sep)[-1].split(".")[0]

        images = convert_from_path(pdf_path, dpi=600, fmt="png")
        #image_paths = []
        for i, page in tqdm(enumerate(images), desc="Converting PDF to Images", unit="page"):
            image_filename = os.path.join(output_dir, f"{pdf_name}_{i+1}.png")
            page.save(image_filename, "PNG")
            #image_paths.append(image_filename)
        

        return output_dir

        
    def get_data(self) -> str:
        """
        Load the images, process them and return a sorted list

        Returns:
            list[str]: sorted list of post-processed image filenames
        """

        has_files = lambda path: os.path.exists(path) and bool(os.listdir(path))

        logger.info("Gathering input data")
        pdf_extracted_path = os.path.join(self.data_path, "extracted_images")

        if self.crop:
            logger.info("Cropping is enabled. Cropped images will be saved in a separate directory")

            logger.info("Checking if cropped images already exist")
            cropped_dir = os.path.join(self.data_path, self.CROPPED_DIR_NAME)
            pdf_cropped_dir = os.path.join(self.data_path, "extracted_images", self.CROPPED_DIR_NAME)
            
            for d in (cropped_dir, pdf_cropped_dir):
                if has_files(d):
                    logger.info(f"Cropped images already exist in {d}. Using them instead of cropping again")
                    return d#natsorted(self.load_files(d))
            
            logger.info("No cropped images found. Cropping the images now")


            images = self.load_files(pdf_extracted_path) if has_files(pdf_extracted_path) else self.load_files() 

            logger.info("Cropping Images...")
            images = self.image_processor(images)

            if not images:
                raise FileNotFoundError(f"No images or PDFs found under {self.data_path}")

            return pdf_cropped_dir if has_files(pdf_extracted_path) else cropped_dir
        else:
            logger.info("Cropping is disabled. Using the original images")
            images = self.load_files(pdf_extracted_path) if has_files(pdf_extracted_path) else self.load_files()

            if not images:
                raise FileNotFoundError(f"No images or PDFs found under {self.data_path}")

            return pdf_extracted_path if has_files(pdf_extracted_path) else self.data_path
            
    
    def __call__(self) -> str:
        """
        Returns the extracted text from the images or loads it from a temp file if it exists

        Returns:
            str: The extracted text
        """
        
        # check for passed in temp extracted text file or default extracted text file. If found, load and return the text
        temp_text_file = os.path.join(self.data_path, self.EXTRACTED_TEXT)
        if not(temp_text_file is None) and os.path.isfile(temp_text_file):
            logger.info("Temperory text file found")
            return self.extraction_model([], temp_text_file)

        # --- Get the best directory given the data path. 
        # if the data path is a directory, return the data path
        # if it is a pdf, load the pdf into images and return the path to the extracted images
        images = self.get_data()

        extracted_text = self.extraction_model(images, None, temp_text_file)

        return extracted_text
