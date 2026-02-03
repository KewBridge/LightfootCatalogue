import os
from lightcat.utils import get_logger
logger = get_logger(__name__)
# OS setting for Pytorch dynamic GPU memory allocation
logger.info("Setting OS environment variables")
logger.info("Setting PYTORCH_CUDA_ALLOC_CONF to expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger.info("Setting TORCH_USE_CUDA_DSA to 1")
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import argparse
from lightcat.data_processing.data_reader import DataReader
from lightcat.models.transcription_model import TranscriptionModel
from lightcat.models.ocr_model import OCRModel
from lightcat.utils.prompt_utils import PromptLoader
from torch.cuda import is_available

logger.info(f"GPU Status: {is_available()}")

def parse_args() -> argparse.Namespace:
    """
    Parses arguments inputted in command line
    
    Flags available:
    -mt or --max-tokens -> for defining maximum number of tokens in the model
    -out or --save-path -> for defining the save path in which to save the jsons
    -b or --batch -> for defining the batch size
    """
    parser = argparse.ArgumentParser(description='Run inference on pages')
    parser.add_argument('images', help='Path to images (Can parse in either a single image or a directory of images)')
    parser.add_argument('prompt', help='Path to an input prompt/conversation to the model')
    parser.add_argument("--ocr-only", action="store_true", help="Only run OCR on the images and save the text to a file")
    parser.add_argument("--test", action="store_true", help="Test mode where testing on only the first 5 images")
    args = parser.parse_args()

    return args


def run_stage_1(images: str, prompt: PromptLoader) -> str:
    """
    Run stage 1 of the pipeline: Data reading and OCR

    Args:
        images (str): the path to either a single image or a directory of images
        prompt (PromptLoader): PromptLoader object containing the loaded prompt

    Returns:
        str: Extracted text from the images
    """

    # Intialise DataReader
    logger.info(">>> Initializing DataReader...")
    data_reader = DataReader(images, prompt=prompt)
    
    # Load the extracted text
    logger.info(">>> Extracting text from images...")
    extracted_text = data_reader()

    return extracted_text


def run_stage_2(prompt: PromptLoader, extracted_text: str) -> None:
    """
    Run stage 2 of the pipeline: Transcription model inference

    Args:
        prompt (PromptLoader): PromptLoader object containing the loaded prompt
        extracted_text (str): Extracted text from the images
    """

    transcription_model = TranscriptionModel(prompt=prompt)

    logger.info(">>> Running Inference...")
    # Perform inference and save the jsons
    _ = transcription_model(extracted_text, save=True, save_file_name=prompt["output_save_file_name"])


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run the full pipeline of data reading, OCR, and transcription model inference

    Args:
        args (argparse.Namespace): Parsed command line arguments
    """

    #============================================
    # Loading prompt and creating output directory
    #============================================
    logger.info(">>> Loading Prompt...")
    prompt = PromptLoader(args.prompt)

    #if not(os.path.exists(prompt["output_save_path"])):
    logger.debug(f"Creating output save path directory at: {prompt['output_save_path']}")
    os.makedirs(prompt["output_save_path"], exist_ok=True)

    #============================================
    # Stage 1: Data Reading and OCR
    #============================================
    extracted_text = run_stage_1(args.images, prompt)

    #============================================
    # If only OCR is required, exit here
    #============================================
    if args.ocr_only:
        logger.info(">>> OCR Finished")
        return

    #============================================
    # Stage 2: Transcription Model Inference
    #===========================================

    if args.test:
        logger.debug("Test mode enabled. Only processing the first/upto 5000 characters")
        extracted_text = extracted_text[:min(5000, len(extracted_text))]
    
    run_stage_2(prompt, extracted_text)

    logger.info(">>> Inference Finished")


def main():
    """
    Main function to perform the operations
    """
    logger.info(">>> Starting...")
    
    args = parse_args()
    logger.info(f"Input arguments: {args}")

    run_pipeline(args)
    
    

if __name__ == "__main__":
    main()
    
