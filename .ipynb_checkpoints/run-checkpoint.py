import argparse
import os
import time
from lib.config import CROPPED_DIR_NAME
import lib.utils as utils
import lib.model as model
import lib.roi as roi



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
    parser.add_argument('-p', '--prompt', default=None, help='Path to an input prompt/conversation to the model')
    parser.add_argument('-mt', '--max-tokens', default=None, help="Maximum number of tokens for model")
    parser.add_argument('-out','--save-path', default=None, help="Save path for json files")
    parser.add_argument('-b','--batch', default=None, help="Batch Size for inference if more than one image provided")
    parser.add_argument('-c', '--crop', default=True, help="Choose to crop and resize an image before parsing into system")
    args = parser.parse_args()

    return args


def main():
    """
    Main function to perform the operations
    """
    print(">>> Starting...")
    args = parse_args()

    # Load a list of all image paths (with their absolute path)
    print(">>> Loading Images...")
    

    cropped_dir = os.path.join(args.images, CROPPED_DIR_NAME)
    
    if args.crop and not(os.path.isdir(cropped_dir)):
        images = utils.load_images(args.images)
        print(">>> Cropping Images...")
        images = roi.cropAllImages(images, pad = 100.0, resize_factor = 0.4, 
              remove_area_perc = 0.01, save_file_name = None)
    elif os.path.isdir(cropped_dir):
        images = utils.load_images(cropped_dir)
    else:
        images = utils.load_images(args.images)

    batch = int(args.batch) if (args.batch is not None) else None
    max_tokens = int(args.max_tokens) if (args.max_tokens is not None) else None
    
    # Load model
    print(">>> Loading Model...")
    qwen_model = model.QWEN_model(prompt= args.prompt, batch_size = batch, max_new_tokens = max_tokens, save_path=args.save_path)
    
    # Perform inference and save the jsons
    _ = qwen_model.batch_infer(images, save=True)
    print(">>> Inference Finished")
    

if __name__ == "__main__":
    main()
    
