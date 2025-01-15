from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import lib.config as config
from lib.utils.utils import debugPrint
from torch.amp import autocast


class QWEN_Model:

    MODEL_NAME = config.MODEL

    def __init__(self, 
                 batch_size: int = 3, # Batch size for inference
                 max_new_tokens: int = 5000, # Maximum number of tokens
                 temperature: float = 0.2, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
                ):

        # Load parameters
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

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
    
        return model


    def _load_processor(self) -> object:
        """
        Loads the pre-processor that is used to pre-process the input prompt and images.
    
        Return:
            processor (object): Returns the loaded pretrained processor for the model.
        """
        processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
    
        return processor

    def __call__(self, conversation: list, images:list=None, debug: bool=False) -> list:
        """
        Performs inference on the given set of images.
    
        Parameters:
            images (list): A set of images to batch inference.
    
        Return:
            output_text (list): A set of model outputs for given set of images.
        """

        # Process the input conversation
        debugPrint("\tProcessing text prompts...", debug)
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        if images is None:
            text_prompts = [text_prompt] if isinstance(text_prompt[0], dict) else text_prompt
            images_opened = None
        else:
            # Get N text_prompts for equal number of images
            text_prompts = [text_prompt] * self.batch_size

            debugPrint("\tReading Images (If available)...", debug)
            # Open the images from the paths if available
            images_opened = [Image.open(image) for image in images]

        # Preprocess the inputs
        debugPrint("\tProcessing inputs...", debug)
        inputs = self.processor(
            text=text_prompts, images=images_opened, padding=True, return_tensors="pt"
        )

        debugPrint("\tMoving inputs to gpu...", debug)
        # Move inputs to device (model automatically moves)
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu") # 
    
        # Inference
        debugPrint("\tPerforming inference...", debug)
        with autocast("cuda"): # Enabling mixed precision to reduce computational load where possible
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens) 
        # Increasing the number of new tokens, increases the number of words recognised by the model with trade-off of speed
        # 1024 new tokens was capable of reading upto 70% of the input image (pg132_a.jpeg)
        debugPrint("\tInference Finished", debug)
        output_ids = output_ids.cpu()

        debugPrint("\tSeperating Ids...", debug)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        # Using the preprocessor to decode the numerical values into tokens (words)
        debugPrint("\tDecoding Ids...", debug)
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )


        return output_text

    
