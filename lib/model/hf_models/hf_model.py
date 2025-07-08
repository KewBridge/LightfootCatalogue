# Python Modules
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
from torch.amp import autocast
import gc
# Logging
from lib.utils import get_logger
logger = get_logger(__name__)

class HF_Model:

    DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    MODEL_TYPE = "multi" # multi if multi-model, single if single model => Used to define the type of prompt to be used

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
        self.model_name = model_name if model_name else self.DEFAULT_MODEL_NAME
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Precompute device: GPU is preferred if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model

        self.model = None
        # Load processor
        
        self.processor = None

    def load(self):
        """
        Load the model and processor.
        This method is called to ensure that the model and processor are loaded before inference.
        """
        if self.model:
            logger.warning("Model is already loaded. Unloading before loading again.")
            del self.model
            self.model = None
            gc.collect()
            
        print(f"Loading model for [{self.model_name}] to device [{self.device}]")
        self.model = self._load_model()

        
        if self.processor:
            logger.warning("Processor is already loaded. Unloading before loading again.")
            del self.processor
            self.processor = None
            gc.collect()
        
        print(f"Loading processor for [{self.model_name}] to device [{self.device}]")
        self.processor = self._load_processor()

        try:
            if self.processor.pad_token is None:
                logger.warning("Pad token is None. Setting pad token to eos token.")
                self.processor.pad_token = self.processor.eos_token
        except Exception as e:
            logger.error(f"Error setting pad token: {e}")

    
    def unload(self):

        del self.model
        del self.processor

        self.model = None
        self.processor = None

        
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self) -> object:
        """
        Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """
        model = AutoModel.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
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
                            images: list[str]=None) -> object:
        """

        Processes the input conversation and images to prepare them for the model.

        Args:
            conversation (list): input prompt to the model
            images (list[str], optional): input images to model. Defaults to None.

        Returns:
            object: A Batch Feature/Ecnoding object containing the processed inputs.
        """

        text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        text_prompts = [text_prompt] if isinstance(text_prompt[0], dict) else text_prompt

        
        if images is None or images == []:
            # If no images are provided, only process text prompts
            inputs = self.processor(
                text=text_prompts,
                return_tensors="pt",
                padding=True,
            )
        else:
            images_opened = [Image.open(image) if isinstance(image, str) else image for image in images]
            inputs = self.processor(
                    text=text_prompts,
                    images=images_opened,
                    return_tensors="pt",
                    padding=True,
                )

        inputs = inputs.to(self.device)
        max_tokens = inputs.input_ids.shape[1]
        return inputs, max_tokens

    def inference_chat_model(self, inputs: object, 
                             max_new_tokens: int = None, 
                             debug: bool = False) -> list:
        """_summary_

        Args:
            inputs (object): _description_
            max_new_tokens (int, optional): _description_. Defaults to None.

        Returns:
            list: _description_
        """

        
        # Inference
        if debug:
            logger.debug("\tPerforming inference...")
        
        do_sample_set = self.temperature > 0.0 # If temperature is greater than 0, use sampling
        with autocast("cuda", enabled=self.device.type == "cuda"): # Enabling mixed precision to reduce computational load where possible
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=self.temperature, do_sample=do_sample_set,)

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
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )


        return output_text


    def _check(self):

        if self.model is None or self.processor is None:
            self.load()


    def __call__(self, conversation: list, images:list[str]=None, 
                 debug: bool=False, max_new_tokens=None) -> list:
        """
        Performs inference on the given set of images and/or text.

        When images are provided, the text is extracted.
        When text is provided, images is set to None and inference is determined by conversation
    
        Parameters:
            conversation (list): The input prompt to the model
            images (list): A set of images to batch inference.
            debug (bool): Used to print debug prompts
            max_new_tokens (int): The maximum number of new tokens to generate. If None, the maximum tokens are used

        Return:
            output_text (list): A set of model outputs for given set of images.
        """

        self._check()

                # Set the device to the model's device
        self.eval()

        # Process the input conversation
        if debug:
            logger.debug("\tProcessing inputs...")

        inputs , max_tokens = self.process_chat_inputs(conversation, images) 
        
        if max_new_tokens is None:
            max_new_tokens = max_tokens   

        output_text = self.inference_chat_model(inputs, max_new_tokens)
        


        return output_text
