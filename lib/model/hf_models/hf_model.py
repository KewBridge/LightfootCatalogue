# Python Modules
import logging
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
from torch.amp import autocast

logger = logging.getLogger(__name__)

class HF_Model:

    DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME, # Model name
                 batch_size: int = 1, # Batch size for inference
                 max_new_tokens: int = 4096, # Maximum number of tokens
                 temperature: float = 0.3, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
                ):
        """
        Hugging Face model class

        This class loads the necessary modules and performs inference given conversation and input

        Parameters:
            model_name (str): Model name
            batch_size (int): batch size for inference
            max_new_tokens (int): Maximum number of tokens
            temperature (float): Model temperature. 0 to 2. Higher the value the more random and
                                 lower the temperature the more focussed and deterministic.
        """

        # Load parameters
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Precompute device: GPU is preferred if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model

        print(f"Loading model for [{self.model_name}] to device [{self.device}]")
        self.model = self._load_model()
        # Load processor
        print(f"Loading processor for [{self.model_name}] to device [{self.device}]")
        self.processor = self._load_processor()


    def _load_model(self) -> object:
        """
        Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """
        model = AutoModel.from_pretrained(
            self.model_name,temperature=self.temperature, torch_dtype="auto", device_map="auto"
        )

        model.gradient_checkpointing_enable()
    
        return model

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()

    def _load_processor(self) -> object:
        """
        Loads the pre-processor that is used to pre-process the input prompt and images.
    
        Return:
            processor (object): Returns the loaded pretrained processor for the model.
        """
        min_pixels = 256*28*28
        max_pixels = 1024*28*28 
        processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    
        return processor
    
    ###################################
    # Processing inputs to chat models
    ###################################


    def process_chat_inputs(self, conversation: list, 
                            images: list[str]=None, 
                            add_padding=True) -> object:
        """

        Processes the input conversation and images to prepare them for the model.

        Args:
            conversation (list): input prompt to the model
            images (list[str], optional): input images to model. Defaults to None.
            add_padding (bool, optional): Whether to add padding to the input text. Defaults to True.

        Returns:
            object: A Batch Feature/Ecnoding object containing the processed inputs.
        """

        text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        text_prompts = [text_prompt] if isinstance(text_prompt[0], dict) else text_prompt

        images_opened = None if not(images) else [Image.open(image) for image in images]

        inputs = self.processor(
            text=text_prompts,
            images=images_opened,
            return_tensors="pt",
            padding=add_padding,
        )

        inputs = inputs.to(self.device)

        return inputs

    def inference_chat_model(self, inputs: object, 
                             max_new_tokens: int = None, 
                             skip_special_tokens: bool = True, debug: bool = False) -> list:
        """_summary_

        Args:
            inputs (object): _description_
            max_new_tokens (int, optional): _description_. Defaults to None.
            skip_special_tokens (bool, optional): _description_. Defaults to True.

        Returns:
            list: _description_
        """

        
        # Inference
        if debug:
            logger.debug("\tPerforming inference...")
        
        
        with autocast("cuda", enabled=self.device.type == "cuda"): # Enabling mixed precision to reduce computational load where possible
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Increasing the number of new tokens, increases the number of words recognised by the model with trade-off of speed
        # 1024 new tokens was capable of reading upto 70% of the input image (pg132_a.jpeg)
        if debug:
            logger.debug("\tInference Finished")

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
            generated_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )


        return output_text


    def __call__(self, **kargs) -> list:
        
        raise NotImplementedError("This method should be implemented in subclasses.")
