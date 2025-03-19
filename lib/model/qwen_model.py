# Python Modules
import logging
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.amp import autocast

# Custom Modules
import lib.config as config


logger = logging.getLogger(__name__)

class QWEN_Model:

    MODEL_NAME = config.MODEL

    def __init__(self, 
                 batch_size: int = 1, # Batch size for inference
                 max_new_tokens: int = 4096, # Maximum number of tokens
                 temperature: float = 0.3, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
                ):
        """
        QWEN model class

        This class loads the necessary modules and performs inference given conversation and input

        Parameters:
            batch_size (int): batch size for inference
            max_new_tokens (int): Maximum number of tokens
            temperature (float): Model temperature. 0 to 2. Higher the value the more random and
                                 lower the temperature the more focussed and deterministic.
        """

        # Load parameters
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Precompute device: GPU is preferred if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model()
        # Load processor
        self.processor = self._load_processor()


    def _load_model(self) -> object:
        """
        Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_NAME,temperature=self.temperature, torch_dtype="auto", device_map="auto"
        )

        model.gradient_checkpointing_enable()
    
        return model


    def _load_processor(self) -> object:
        """
        Loads the pre-processor that is used to pre-process the input prompt and images.
    
        Return:
            processor (object): Returns the loaded pretrained processor for the model.
        """
        min_pixels = 256*28*28
        max_pixels = 1024*28*28 
        processor = AutoProcessor.from_pretrained(self.MODEL_NAME, min_pixels=min_pixels, max_pixels=max_pixels)
    
        return processor
    

    def __call__(self, conversation: list, images:list[str]=None, debug: bool=False) -> list:
        """
        Performs inference on the given set of images and/or text.

        When images are provided, the text is extracted.
        When text is provided, images is set to None and inference is determined by conversation
    
        Parameters:
            conversation (list): The input prompt to the model
            images (list): A set of images to batch inference.
            debug (bool): Used to print debug prompts
    
        Return:
            output_text (list): A set of model outputs for given set of images.
        """

        # Process the input conversation
        if debug:
            logger.debug("\tProcessing text prompts...")
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        if images is None:
            text_prompts = [text_prompt] if isinstance(text_prompt[0], dict) else text_prompt
            images_opened = None
        else:
            # Get N text_prompts for equal number of images
            text_prompts = [text_prompt] * self.batch_size

            if debug:
                logger.debug("\tReading Images (If available)...")
            # Open the images from the paths if available
            images_opened = [Image.open(image) for image in images]

        # Preprocess the inputs
        if debug:
            logger.debug("\tProcessing inputs...")
        inputs = self.processor(
            text=text_prompts, images=images_opened, padding=True, return_tensors="pt"
        )


        if debug:
            logger.debug("\tMoving inputs to gpu...")
        # Move inputs to device (model automatically moves)
        inputs = inputs.to(self.device.type) # 
    
        # Inference
        if debug:
            logger.debug("\tPerforming inference...")
        
        with autocast("cuda", enabled=self.device.type == "cuda"): # Enabling mixed precision to reduce computational load where possible
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # Increasing the number of new tokens, increases the number of words recognised by the model with trade-off of speed
        # 1024 new tokens was capable of reading upto 70% of the input image (pg132_a.jpeg)
        if debug:
            logger.debug("\tInference Finished")
        output_ids = output_ids.cpu()

        if debug:
            logger.debug("\tSeperating Ids...")
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        # Using the preprocessor to decode the numerical values into tokens (words)
        if debug:
            logger.debug("\tDecoding Ids...")
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )


        return output_text

    
