# Python Modules
import os
from tqdm import tqdm
from typing import Optional, Union
import re
import cv2
from pytesseract import image_to_string
import numpy as np
import gc
from PIL import Image

# Import Custom Modules
from lib.utils.file_utils import save_to_file, load_from_file, get_save_file_name
from lib.model import get_model
from lib.model.base_model import BaseModel
from lib.utils.promptLoader import PromptLoader
from lib.utils.save_utils import save_json, save_csv_from_json, verify_json
from lib.data_processing.text_processing import TextProcessor
from lib.data_processing.chunker import TextChunker

# Logging
from lib.utils import get_logger
logger = get_logger(__name__)

class OCRModel(BaseModel):
    
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
            **kwargs (dict): extra parameters for other models
        """
        super().__init__(prompt, **kwargs)
        self.temperature = prompt["ocr_temperature"] if prompt["ocr_temperature"] is not None else 0.
        self.overlap = 100
        self.context_size = 500
        self.model_name = prompt["ocr_model"]
        self.extraction_model = None
        self.extraction_model_name = prompt["ocr_extraction_model"]
        self.grayscale_only = prompt["grayscale_only"]
        self.model = self.load_model()
        self.chunker = TextChunker(self.overlap, prompt["max_chunk_size"])

    def getImagePrompt(self) -> list:
        """
        Get the image extraction prompt

        Returns:
            list: Image extraction conversation to VL model
        """

        system_prompt = (
            "You are an expert in extracting text from images."
        )

        image_prompt = (
            "Extract only the main body text from the image, preserving the original structure and formatting. \n"
            "Do not perform any grammatical corrections. Ignore Page numbers and any other text that is not part of the main body text.\n"
            "Do not generate any additional text or explanations."
        )

        return self.prompt.getImagePrompt(system_prompt, image_prompt)
    

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
            "You are an expert in cleaning OCR induced errors in the text. \n"
            "Follow the instructions below to clean the text, ensuring the text flows coherently with the previous context:\n"
            "1. Fix OCR induced typographical errors, such as incorrect characters, spacing and improper symbols.\n"
            "- Use provided context and common sense to identify and correct errors.\n"
            "- The letter 'AE' and 'Æ' are often confused with symbols such as '&' and other special symbols.\n"
            "- For example, 'l' and '1' or 'o' and '0' are often confused.\n"
            "- Ensure that the text is grammatically correct and coherent.\n"
            "- Remove any unnecessary line breaks or extra spaces.\n"
            "- Identify and correct word splits and line breaks.\n"
            "- Only fix clear OCR errors. DO NOT ALTER THE CONTEXT OR MEANING of the text.\n"
            "- DO NOT add any generated text, punctuation, or capitalization.\n"
            "2. Ensure structure is maintained.\n"
            "- Maintain original structure, including paragraphs and line breaks.\n"
            "- Preserve the original content. \n"
            "- Keep all importatnt information intact.\n"
            "- DO NOT add any new text not present in the text. \n"
            "3. Ensure flow and coherence.\n"
            "- Ensure the text flows naturally and coherently.\n"
            "- Use provided context to ensure the text makes sense.\n"
            "- HANDLE text that starts or ends mid-sentence correctly. \n\n"
            "4. Return ONLY the cleaned text.\n"
            "- Do not add any additional information, explanations, or thoughts.\n"
            "- Do not include your thoughts, explanations, or steps.\n"
            "- Do not add any new text not present in the text.\n"
        )
        noise_prompt = (
            # "IMPORTATANT: RETURN ONLY THE CLEANED TEXT. Preserve the orignial structure and content. Do not add anything else. Do not include your thoughts, explantions or steps.\n\n"
            f"Previous context:\n {context}\n\n"
            f"Text to clean:\n {extracted_text}\n\n"
            "Cleaned text:\n"
        )

        return self.prompt.getTextPrompt(system_prompt, noise_prompt)

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
        
        ocr_cleaning_prompt = self.getOCRNoiseCleaningPrompt(text)
        cleaned_text = self.model(ocr_cleaning_prompt)
        cleaned_text = self.clean(cleaned_text[0])

        # seperation_prompt = self.getSeperateRecordsPrompt(cleaned_text)
        # organised_text = self.model(seperation_prompt)
        # cleaned_text = self.clean(organised_text[0])

        cleaned_text_split = cleaned_text.split("\n")

        # Return the cleaned text and the last record
        return cleaned_text #"\n".join(cleaned_text_split[:-1]), cleaned_text_split[-1]
    

    def detect_column_cuts(self, gray_img, min_gap_width=50, gap_thresh_ratio=0.05):
        """
        Given a grayscale page image, return the x-coordinates where you should cut 
        to split into columns.  If no good gutter is found, returns [] (i.e. one column).
        """
        # 1) Binarize (invert so text is “1”)
        binarized_image = cv2.threshold(gray_img, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # 2) Optional: close tiny text-break gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        binarized_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, kernel)
        # 3) Sum ink pixels in each column
        vert_proj = binarized_image.sum(axis=0)  # shape: (width,)
        # 4) Threshold to find “mostly white” columns
        thresh = vert_proj.max() * gap_thresh_ratio
        is_gap = vert_proj < thresh
        # 5) Find contiguous gap runs wider than min_gap_width
        cuts = []
        start = None
        for x, val in enumerate(is_gap):
            if val and start is None:
                start = x
            elif not val and start is not None:
                width = x - start
                if width >= min_gap_width:
                    cuts.append((start + x)//2)  # mid-point of the gap
                start = None
        # handle case run-to-end
        if start is not None and (len(is_gap) - start) >= min_gap_width:
            cuts.append((start + len(is_gap))//2)
        return cuts
    

    def single_image_extract(self, image: Union[str, np.ndarray]) -> str:

        logger.debug("Extracting text from image...")
        logger.debug(f"Image type: {type(image)}")
        logger.debug(f"Image shape: {image.shape if isinstance(image, np.ndarray) else 'N/A'}")
        logger.debug(f"Image dtype: {image.dtype if isinstance(image, np.ndarray) else 'N/A'}")
        logger.debug("=" * 50)

        if isinstance(image, np.ndarray):

            image = Image.fromarray(image)
        text = ""
        if (self.extraction_model_name is None) or (self.extraction_model_name.lower() == "default"):
            text = image_to_string(image, lang="eng+lat", config="--psm 4")
        else:
            print(f"Using {self.extraction_model_name} for text extraction.")
            if self.extraction_model is None:
                self.extraction_model = self.load_model(self.extraction_model_name)
            message = self.getImagePrompt()
            text = self.extraction_model(message, [image])[0]
        
        return text.strip()
        
    def ocr_page_with_columns(self, path_to_image: Union[str, np.ndarray]) -> str:

        if isinstance(path_to_image, str):
            img = cv2.imread(path_to_image)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(path_to_image, np.ndarray):
            grey_img = path_to_image
        else:
            raise ValueError("Input must be a file path or a numpy array.")
            
        cuts = self.detect_column_cuts(grey_img)

        # define our column boxes
        xs = [0] + cuts + [grey_img.shape[1]]
        text = []
        
        #col_imgs = []
        for left, right in zip(xs[:-1], xs[1:]):
            col_img = grey_img[:, left:right]
            
            #col_imgs.append(col_img)
            txt = self.single_image_extract(col_img)
            text.append(txt.strip())
        # join them in reading order, separated by two line breaks
        return "\n\n\n\n\n".join(text)
    
    def extract_text_from_image(self, image:list[np.ndarray]) -> list[str]:
        """
        Extract text from a single image

        Parameters:
            image (str): path to the image

        Returns:
            list: extracted text from the image
        """
        
        
        # Load the image
        return [self.ocr_page_with_columns(im) if self.prompt["has_columns"] else self.single_image_extract(im) for im in image]
    

    def clean_text(self, text):
        
        text = re.sub(r"\n{2,}", "\n\n", text)  # Remove excessive newlines
        
        text = text.strip()
        # Clean line breaks
        text = re.sub(r"-\n+", "", text)
        text = re.sub(r"^(\d*|[a-z]{0,3})\s?(Joh|Cat).*\n", "", text, re.I)
        text = re.sub(r"^.*(l?o?gue|ghtfoot|foot)\s?(\d*)?\n", "", text, re.I)
        text = re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", text)
        text = re.sub(r"([A-Z]+)\s*(EAE|FAE|EAF)", r"\1EAE", text)
        
        # Add a newline before any indexing patterns like 1. or i. or a., but only if not part of a word ending
        # Ensure the pattern is preceded by whitespace or start of line, and not a letter (to avoid word endings)
        # exclusions = r'(?:e\.g\.|i\.e\.|etc\.|cf\.|vs\.)'
        # text = re.sub(rf'(?<![a-zA-Z0-9])\s+(?!{exclusions})(\d+\.)\n?', r'\n\1', text)
        # text = re.sub(rf'(?<![a-zA-Z0-9])\s+(?!{exclusions})([ivxlc]+\.)\n?', r'\n\1', text)
        # text = re.sub(rf'(?<![a-zA-Z0-9])\s+(?!{exclusions})([a-z]\.)\n?', r'\n\1', text)
        
        #text = re.sub(r"\n\n+", "\n\n", text)
        return text
    

    def extract_text(self, images: list[str], save_file: str = None, debug: bool = False, clean: bool = True) -> str:
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

        #Add previous block of text to the next batch
        # This is done to ensure that the model does not forget the previous text
        # Add tqdm for progress tracking
        for ind, batch in enumerate(tqdm(images, desc="Processing images", unit="image")):
            #print(f">>> Batch {ind + 1} starting...")

            if debug:
                logger.debug("Extracting text from image")

            extracted_text = self.extract_text_from_image([batch])

            cleaned_text = [self.clean_text(text) for text in extracted_text] if clean else extracted_text

            if debug:
                logger.debug("\tJoining Outputs...")
            # Join all the text and append it together with previous batches
            batch_texts.append("\n\n".join(cleaned_text))

            if (ind + 1) % self.SAVE_TEXT_INTERVAL == 0:
                if debug:
                    logger.debug("\tStoring at interval...")
                save_to_file(self.TEMP_TEXT_FILE if save_file is None else save_file, "\n\n".join(batch_texts))

            if debug:
                logger.debug("\tBatch Finished")

        return "".join(batch_texts) 


    def process_images(self, images: list[str], grayscale_only:bool = False) -> np.ndarray:
        
        logger.debug("Step 1: Grayscale Conversion")
        gray_images = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY) for image in images]

        if grayscale_only or self.grayscale_only:
            return gray_images
        logger.debug("Step 2: Remove shadows")
        thresh = lambda x: cv2.adaptiveThreshold(x, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)

        median_filter = lambda x: cv2.medianBlur(x, 3)
        shadows_removes = [median_filter(thresh(image)) for image in gray_images]

        logger.debug("Step 3: Noise Reductiion")
        denoised_images = [cv2.bilateralFilter(image, 9, 75, 75) for image in shadows_removes]


        logger.debug("Setp 4: Binarization (black and White) via Otsu's method")
        binarized_images = [cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for image in denoised_images]


        logger.debug("Step 5: Deskewing")
        logger.debug("Skipping deskewing for now")
        deskewed_images = binarized_images

        logger.debug("Step 6: Morphological opening (erosion followed by dilation)")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))#np.ones((1, 1), np.uint8)
        opened = lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN,
                                            kernel, iterations=1)
        opened_images = [opened(image) for image in deskewed_images]


        logger.debug("Step 7: Optional dilation to thicken strokes")
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        processed = [cv2.dilate(i, kernel2, iterations=1) for i in opened_images]

        return processed
    

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

            chunk = re.sub(r"\[", "\[", chunk)
            chunk = re.sub(r"\]", "\]", chunk)
            
            message = self.getOCRNoiseCleaningPrompt(chunk, context)

            cleaned_chunk = self.model(message)[0]

            cleaned_chunks.append(cleaned_chunk)

            context = cleaned_chunk[-self.context_size:] if len(cleaned_chunk) > self.context_size else cleaned_chunk
            gc.collect()

        merged_text = self.chunker.merge_sentences(cleaned_chunks)
        return merged_text


    def __call__(self, images: list[str], text_file: Optional[str] = None, save_file: str = None, debug: bool = False) -> str:
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

        
        self.info()

        
        if text_file is None:

            logger.info(f"Processing input images before extraction...")
            images = self.process_images(images)
            logger.info(f"Extracting text from images...")
            extracted_text = self.extract_text(images, save_file, debug)
            logger.info(f"Chunking text for cleaning...")
            extracted_text = self.chunk_and_clean(extracted_text, add_overlap=True)

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
