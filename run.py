import argparse
import os
import time
from lib.config import CROPPED_DIR_NAME
import lib.utils.utils as utils
import lib.pages.roi as roi
from lib.model.base_model import BaseModel

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
    parser.add_argument('save-file-name', help="Save file name for the outputs")
    parser.add_argument('temp-text', default=None, help="Temporary file storing the extracted text")   
    parser.add_argument('-mt', '--max-tokens', default=100000, help="Maximum number of tokens for model")
    parser.add_argument('--max-chunk-size', default=3000, help="Define the maximum size of each text block")
    parser.add_argument('--save-path', default=None, help="Save path for json files")
    parser.add_argument('-b','--batch', default=None, help="Batch Size for inference if more than one image provided")
    parser.add_argument('-c', '--crop', default=True, help="Choose to crop and resize an image before parsing into system")
    args = parser.parse_args()

    return args

def get_images(image_path: str, crop: bool = True, pad: float = 100.0, resize_factor: float = 0.4, remove_area_perc: float = 0.01, save_file_name: str = None):
    print(">>> Loading Images...")
    

    cropped_dir = os.path.join(image_path, CROPPED_DIR_NAME)
    
    if crop and not(os.path.isdir(cropped_dir)):
        images = sorted(utils.load_images(image_path))
        print(">>> Cropping Images...")
        images = roi.cropAllImages(images, pad, resize_factor, 
              remove_area_perc, save_file_name)
    elif os.path.isdir(cropped_dir):
        images = utils.load_images(cropped_dir)
    else:
        images = utils.load_images(image_path)

    return images

def main():
    """
    Main function to perform the operations
    """
    print(">>> Starting...")
    args = parse_args()

    # Load a list of all image paths (with their absolute path)
    images = get_images(args.images, args.crop)

    batch = int(args.batch) if (args.batch is not None) else None
    max_tokens = int(args.max_tokens) if (args.max_tokens is not None) else None
    
    # Load model
    print(">>> Loading Model...")
    model = BaseModel("qwen_model", prompt=args.prompt, max_new_tokens = max_tokens, batch_size=batch, temperature=1, save_path=args.save_path)
    #qwen_model = model.QWEN_model(prompt= args.prompt, batch_size = batch, max_new_tokens = max_tokens, save_path=args.save_path)
    
    # Perform inference and save the jsons
    _ = model(images, args.temp_text, save=True, save_file_name=args.save_file_name)
    print(">>> Inference Finished")
    

if __name__ == "__main__":
    main()
    
