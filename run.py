import os
import logging

logger = logging.getLogger(__name__)
# OS setting for Pytorch dynamic GPU memory allocation
logger.info("Setting OS envrionmnet variables")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import argparse
from lib.data_processing.data_reader import DataReader
from lib.model.base_model import BaseModel
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
    parser.add_argument('savefilename', help="Save file name for the outputs")
    parser.add_argument('--temp-text', default=None, help="Temporary file storing the extracted text")   
    parser.add_argument('-mt', '--max-tokens', default=100000, help="Maximum number of tokens for model")
    parser.add_argument('--max-chunk-size', default=2800, help="Define the maximum size of each text block")
    parser.add_argument('--save-path', default=None, help="Save path for json files")
    parser.add_argument('-b','--batch', default=1, help="Batch Size for inference if more than one image provided")
    parser.add_argument('-c', '--crop', default=True, help="Choose to crop and resize an image before parsing into system")
    parser.add_argument('--debug', default=False, help="Debug mode where testing on only the first 5 images")
    args = parser.parse_args()

    return args

def main():
    """
    Main function to perform the operations
    """
    logger.info(">>> Starting...")
    
    args = parse_args()
    logger.info(f"Input arguments: {args}")

    # Load model
    logger.info(">>> Loading Model...")

    batch = int(args.batch) if (args.batch is not None) else None
    max_tokens = int(args.max_tokens) if (args.max_tokens is not None) else None

    model = BaseModel("qwen_model", prompt=args.prompt, max_new_tokens = max_tokens, 
                      batch_size=batch, temperature=0.6, save_path=args.save_path)
    
    # Intialise DataReader
    data_reader = DataReader(args.images,model,args.crop,
                             pad = 100.0, resize_factor = 0.4, remove_area_perc = 0.01, save_file_name = None)
    # Load the extracted text
    extracted_text = data_reader(args.temp_text)
    
    #qwen_model = model.QWEN_model(prompt= args.prompt, batch_size = batch, max_new_tokens = max_tokens, save_path=args.save_path)
    
    # Perform inference and save the jsons
    _ = model(extracted_text, save=True, save_file_name=args.savefilename)
    logger.info(">>> Inference Finished")
    

if __name__ == "__main__":
    main()
    
