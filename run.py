import argparse
import os
import time
import lib.config as config
import lib.utils as utils
import lib.model as model




def parse_args():
    """
    Parses arguments inputted in command line
    
    Flags available:
    """
    parser = argparse.ArgumentParser(description='Run inference on pages')
    parser.add_argument('images', help='Path to images (Can parse in either a single image or a directory of images)')
    parser.add_argument('-mt', '--max-tokens', default=None, help="Maximum number of tokens for model")
    parser.add_argument('-out','--save-path', default=None, help="Save path for json files")
    parser.add_argument('-b','--batch', default=None, help="Batch Size for inference if more than one image provided")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    #Load a list of all image paths (with their absolute path)
    images = utils.load_images(args.images)

    #Load model
    qwen_model = model.load_model()
    #Load processor
    processor = model.load_processor()

    #Perform inference
    image_text_pairs = model.batch_infer(qwen_model, processor, images, args.batch, args.max_tokens)
    
    utils.save_jsons(image_text_pairs, args.save_path)
    

if __name__ == "__main__":
    main()
    