from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import lib.config as config


def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    return model


def load_processor():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    return processor


def perform_inference(model: object, processor: object, images : list, max_new_tokens : int = None) -> list:

    #Preprocess the inputs
    max_new_tokens = max_new_tokens if (max_new_tokens is not None) else config.MAX_NEW_TOKENS
    #Prepreocess the conversation
    text_prompt = processor.apply_chat_template(config.CONVERSATION, add_generation_prompt=True)
    #Get N text_prompts for equal number of images
    text_prompts = [text_prompt] * len(images)

    images_opened = [Image.open(image) for image in images]
    #Preprocess the inputs
    inputs = processor(
        text=text_prompts, images=images_opened, padding=True, return_tensors="pt"
    )
    #Move inputs to device (model automatically moves)
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu") # 

    #Inference
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens) 
    ## Increasing the number of new tokens, increases the number of words recognised by the model with trade-off of speed
    ## 1024 new tokens was capable of reading upto 70% of the input image (pg132_a.jpeg)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text

def batch_infer(model: object, processor: object, images: list, batch_size: int = None, max_new_tokens: int = None) -> list:
    max_new_tokens = max_new_tokens if (max_new_tokens is not None) else config.MAX_NEW_TOKENS
    batch_size = batch_size if (batch_size is not None) else config.BATCH_SIZE

    
    batched_images = [images[x:min(x+batch_size, len(images))] for x in range(0, len(images), batch_size)]

    image_text_pairs = []

    for batch in batched_images:

        output_text = perform_inference(model, processor, batch, max_new_tokens)

        pairs = [(image, output_text[ind]) for ind, image in enumerate(batch)]

        image_text_pairs.extend(pairs)

    return image_text_pairs
        

    

    