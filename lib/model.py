from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import lib.config as config
import lib.utils as utils

def load_model() -> object:
    """
    Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).

    Return:
        model (object): Returns the loaded pretrained model.
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.MODEL, torch_dtype="auto", device_map="auto"
    )

    return model


def load_processor() -> object:
    """
    Loads the pre-processor that is used to pre-process the input prompt and images.

    Return:
        processor (object): Returns the loaded pretrained processor for the model.
    """
    processor = AutoProcessor.from_pretrained(config.MODEL)

    return processor


def perform_inference(model: object, processor: object, images : list, max_new_tokens : int = None) -> list:
    """
    Performs inference on the given set of images.

    Parameters:
        model (object): The pretrained model.
        processor (object): The pretrained pre-processor.
        images (list): A set of images to batch inference.
        max_new_tokens (int): Maximum number of tokens (words) for the model to remember/learn.[Default: None]

    Return:
        output_text (list): A set of model outputs for given set of images.
    """
    # Preprocess the inputs
    max_new_tokens = max_new_tokens if (max_new_tokens is not None) else config.MAX_NEW_TOKENS
    # Prepreocess the conversation
    text_prompt = processor.apply_chat_template(config.CONVERSATION, add_generation_prompt=True)
    # Get N text_prompts for equal number of images
    text_prompts = [text_prompt] * len(images)

    images_opened = [Image.open(image) for image in images]
    # Preprocess the inputs
    inputs = processor(
        text=text_prompts, images=images_opened, padding=True, return_tensors="pt"
    )
    # Move inputs to device (model automatically moves)
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu") # 

    # Inference
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens) 
    # Increasing the number of new tokens, increases the number of words recognised by the model with trade-off of speed
    # 1024 new tokens was capable of reading upto 70% of the input image (pg132_a.jpeg)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    # Using the preprocessor to decode the numerical values into tokens (words)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text


def batch_infer(model: object, processor: object, images: list, batch_size: int = None, max_new_tokens: int = None, save_path: str = None) -> list:
    """
    Perform batch inference on a list of images given a batch size

    Parameters:
        model (object): The pretrained model.
        processor (object): The pretrained pre-processor.
        images (list): A set of images to batch inference.
        batch_size (int): The size of a single batch to inference. [Default: None]
        max_new_tokens (int): Maximum number of tokens (words) for the model to remember/learn. [Default: None]

    Return:
        image_text_pairs (list): A list of tuples containing (image, output_text) pairs
    """

    # Define the value for maximum number of tokens and batch size given they are empty (None)
    max_new_tokens = max_new_tokens if (max_new_tokens is not None) else config.MAX_NEW_TOKENS
    batch_size = batch_size if (batch_size is not None) else config.BATCH_SIZE
    print(f" Using: \n Maximum new tokens = {max_new_tokens} \n Batch size = {batch_size} \n save_path = {save_path}")

    # Seperate the input images into batches (return the rest as a single batch)
    # Such that given 10 images and a batch size of 3
    # We get 3 batches of size 3 and a final batch of size 1
    batched_images = [images[x:min(x+batch_size, len(images))] for x in range(0, len(images), batch_size)]

    # Loop through batched images and perform inference
    for ind, batch in enumerate(batched_images):
        print(f"Batch {ind+1} starting...")
        output_text = perform_inference(model, processor, batch, max_new_tokens)
        
        pairs = [(image, output_text[ind]) for ind, image in enumerate(batch)]

        # Saving the output as soon as we get it for each batch, as not to waste memory
        print(f"== Saving Pairs For Batch {ind+1} ==")
        utils.save_jsons(pairs, save_path)
    
        

    

    
