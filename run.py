import os
from lib.utils import get_logger
logger = get_logger(__name__)
# OS setting for Pytorch dynamic GPU memory allocation
logger.info("Setting OS envrionmnet variables")
logger.info("Setting PYTORCH_CUDA_ALLOC_CONF to expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger.info("Setting TORCH_USE_CUDA_DSA to 1")
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import argparse
from lib.data_processing.data_reader import DataReader
from lib.model.transcription_model import TranscriptionModel
from lib.model.ocr_model import OCRModel
from lib.utils.promptLoader import PromptLoader
from torch.cuda import is_available

logger.info(f"GPU Status: {is_available()}")

def parse_args():
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

def main():
    """
    Main function to perform the operations
    """
    logger.info(">>> Starting...")
    
    args = parse_args()
    logger.info(f"Input arguments: {args}")

    logger.info(">>> Loading Prompt...")
    prompt = PromptLoader(args.prompt)

    #if not(os.path.exists(prompt["output_save_path"])):
    logger.debug(f"Creating output save path directory at: {prompt['output_save_path']}")
    os.makedirs(prompt["output_save_path"], exist_ok=True)

    # Intialise DataReader
    logger.info(">>> Initializing DataReader...")
    data_reader = DataReader(args.images, prompt=prompt)
    
    # Load the extracted text
    logger.info(">>> Extracting text from images...")
    extracted_text = data_reader()

    if args.test:
        logger.debug("Test mode enabled. Only processing the first/upto 5000 characters")
        extracted_text = extracted_text[:min(5000, len(extracted_text))]
    
    if args.ocr_only:
        logger.info(">>> OCR Finished")
        return
    # Load the transcription model

    transcription_model = TranscriptionModel(prompt=prompt)

    logger.info(">>> Running Inference...")
    # Perform inference and save the jsons
    _ = transcription_model(extracted_text, save=True, save_file_name=prompt["output_save_file_name"])

    #logger.info(f"Unloading model: {transcription_model.model_name}")
    #transcription_model.model.unload()
    logger.info(">>> Inference Finished")
    

if __name__ == "__main__":
    main()
    
