import os
from lib.utils import get_logger
logger = get_logger(__name__)
# OS setting for Pytorch dynamic GPU memory allocation
logger.info("Setting OS envrionmnet variables")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
    parser.add_argument('--savefilename', help="Save file name for the outputs")
    parser.add_argument('--model', help="Model name to be used")
    parser.add_argument('--temp-text',  help="Temporary file storing the extracted text")   
    parser.add_argument("--ocr-only", action="store_true", help="Only run OCR on the images and save the text to a file")
    #parser.add_argument('--debug', default=False, help="Debug mode where testing on only the first 5 images")
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

    prompt = PromptLoader(args.prompt)

    if args.model:
        prompt["model"] = args.model
    if args.savefilename:
        prompt["output_save_file_name"] = args.savefilename
    

    if not(os.path.exists(prompt["output_save_path"])):
        os.makedirs(args.save_path)

    # Load model
    logger.info(">>> Loading Model...")

    ocr_model = OCRModel(prompt=prompt)

    # Load the model
    print(f"Loading model: {ocr_model.model_name}")
    ocr_model.model.load()
    # Intialise DataReader
    data_reader = DataReader(args.images, extraction_model=ocr_model, prompt=prompt)
    # Load the extracted text
    extracted_text = data_reader(args.temp_text)

    if args.test:
        print("Test mode enabled. Only processing the first/upto 1000 characters")
        extracted_text = extracted_text[:min(1000, len(extracted_text))]
    

    print(f"Unloading model: {ocr_model.model_name}")
    ocr_model.model.unload()
    if args.ocr_only:
        logger.info(">>> OCR Finished")
        return
    # Load the transription model


    transcription_model = TranscriptionModel(prompt=prompt)
    print(f"Loading model: {transcription_model.model_name}")
    transcription_model.model.load()
    # Perform inference and save the jsons
    _ = transcription_model(extracted_text, save=True, save_file_name=prompt["output_save_file_name"])

    print(f"Unloading model: {transcription_model.model_name}")
    transcription_model.model.unload()
    logger.info(">>> Inference Finished")
    

if __name__ == "__main__":
    main()
    
