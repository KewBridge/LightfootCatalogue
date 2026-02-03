# Python Modules
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
from torch.amp import autocast
import gc
# Logging
from lightcat.utils import get_logger
logger = get_logger(__name__)

class HF_Model:

    DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    MODEL_TYPE = "multi" # multi if multi-model, single if single model => Used to define the type of prompt to be used
    CONTEXT_LENGTH = None # Default context length for the model

    def __init__(self,
                 model_name: str = None, # Model name
                 batch_size: int = 1, # Batch size for inference
                 max_new_tokens: int = 4096, # Maximum number of tokens
                 temperature: float = 0.1, # Model temperature. 0 to 2. Higher the value the more random and lower the value the more focused and deterministic.
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
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
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

        if self.model or self.processor:
            logger.warning("Model or Processor is already loaded. Unloading before loading again.")
            self.unload()

        print(f"Loading model for [{self.model_name}] to device [{self.device}]")
        self.model = self._load_model()
        
        print(f"Loading processor for [{self.model_name}] to device [{self.device}]")
        self.processor = self._load_processor()

        try:
            # Ensure left-padding for decoder-only architectures
            if hasattr(self.processor, "padding_side"):
                self.processor.padding_side = "left"
                logger.info("Set processor padding_side to 'left'.")

            # Some processors expose tokenizer instead of pad_token directly
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
                logger.info("Set tokenizer padding_side to 'left'.")

            # Only set pad_token if the attribute exists
            if hasattr(self.processor, "pad_token") and self.processor.pad_token is None:
                if hasattr(self.processor, "eos_token"):
                    logger.warning("Pad token is None. Setting pad token to eos token.")
                    self.processor.pad_token = self.processor.eos_token
            elif tokenizer is not None and hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
                if hasattr(tokenizer, "eos_token"):
                    logger.warning("Tokenizer pad token is None. Setting pad token to eos token.")
                    tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Error configuring processor: {e}")

    
    def unload(self):
        """
        unload the model and processor.
        This method is called to ensure that the model and processor are unloaded after usage.
        """

        # Delete the references stored in the variable
        del self.model
        del self.processor

        # Set the variables to None
        self.model = None
        self.processor = None
        
        # Clear the GPU and CPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_context_length(self):

        """
        Load the context length of the model.

        Return:
            context_length (int): Returns the context length of the model.
        """

        try:
            self.CONTEXT_LENGTH = self.model.config.max_position_embeddings
            logger.info(f"Model context length set to {self.CONTEXT_LENGTH} tokens.")
        except Exception as e:
            logger.error(f"Error loading context length: {e}")

            if self.CONTEXT_LENGTH:
                logger.info(f"Using predefined context length of {self.CONTEXT_LENGTH} tokens.")
            else:
                self.CONTEXT_LENGTH = 4096 # Default to 4096 if unable to load
                logger.info(f"Model context length set to default {self.CONTEXT_LENGTH} tokens.")



    def _load_model(self) -> object:
        """
        Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """
        model = AutoModel.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )

        self._load_context_length()
    
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

        processor = AutoProcessor.from_pretrained(self.model_name)
    
        return processor
    
    ###################################
    # Processing inputs to chat models
    ###################################


    def _load_images(self, images: list[str]) -> list[Image.Image]:
        """
        Loads the images either from paths or return previously opened images

        Args:
            images (list[str]): Input image list. Can either be a list of paths or a list of PIL Images.

        Returns:
            list[Image.Image]: A list of opened PIL Images.
        """

        opened_images = []
        for img in images:
            if isinstance(img, str):
                try:
                    with Image.open(img) as opened_img:
                        opened_images.append(opened_img.copy())
                except Exception as e:
                    logger.error(f"Error opening image {img}: {e}")
            elif isinstance(img, Image.Image):
                opened_images.append(img)
            else:
                raise ValueError(f"Invalid image type: {type(img)}. Must be str or PIL.Image.Image")

        return opened_images


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

        # Support single conversation or batch of conversations
        if conversation and isinstance(conversation[0], list):
            text_prompts = [
                self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversation
            ]
        else:
            text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text_prompts = [text_prompt] if isinstance(text_prompt, str) else text_prompt

        
        if images:
            images_opened = self._load_images(images)
            inputs = self.processor(
                    text=text_prompts,
                    images=images_opened,
                    return_tensors="pt",
                    padding=True,
                )
        else:
            # If no images are provided, only process text prompts
            inputs = self.processor(
                    text=text_prompts,
                    return_tensors="pt",
                    padding=True,
                )

        inputs = inputs.to(self.device)
        max_tokens = inputs.input_ids.shape[1]
        return inputs, max_tokens


    def inference_chat_model(self, inputs: object,
                             max_new_tokens: int = 4096, 
                             debug: bool = False) -> list:
        """
        Performs inference on the chat model given the processed inputs.

        Args:
            inputs (object): Processed inputs for the model.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 4096.
            debug (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            list: A list of generated text outputs from the model.
        """

        
        # Inference
        logger.debug("\tPerforming inference...")
        
        enable_mixed_precision = self.device.type == "cuda"
        with autocast("cuda", enabled=enable_mixed_precision): # Enabling mixed precision to reduce computational load where possible
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=self.temperature)

        # Increasing the number of new tokens, increases the number of words recognised by the model with trade-off of speed
        # 1024 new tokens was capable of reading upto 70% of the input image (pg132_a.jpeg)
        logger.debug("\tInference Finished")

        logger.debug("\tSeperating Ids...")
        generated_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]

        # Using the preprocessor to decode the numerical values into tokens (words)
        logger.debug("\tDecoding Ids...")
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        return output_text


    def _check(self):

        if self.model is None or self.processor is None:
            self.load()


    def __call__(self, conversation: list, images:list[str]=None, 
                 debug: bool=False, return_input_image_size=False, **kwargs) -> list:
        """
        Performs inference on the given set of images and/or text.

        When images are provided, the text is extracted.
        When text is provided, images is set to None and inference is determined by conversation
    
        Parameters:
            conversation (list): The input prompt to the model
            images (list): A set of images to batch inference.
            debug (bool): Used to print debug prompts
            return_input_image_size (bool): Whether to return the input image size along with output text. Default is False.

        Return:
            output_text (list): A set of model outputs for given set of images.
        """

        self._check()

                # Set the device to the model's device
        self.eval()

        # Process the input conversation
        logger.debug("\tProcessing inputs...")

        inputs , max_tokens = self.process_chat_inputs(conversation, images)
        
        new_max_tokens = self.max_new_tokens

        if max_tokens + self.max_new_tokens > self.CONTEXT_LENGTH:

            new_max_tokens = self.CONTEXT_LENGTH - max_tokens

            if new_max_tokens <= 0:

                raise ValueError(f"Total tokens ({max_tokens + self.max_new_tokens}) exceed model context length ({self.CONTEXT_LENGTH}). Consider reducing max_new_tokens or input length.\n"
                                f"Current max_new_tokens: {self.max_new_tokens}, Input tokens: {max_tokens}, Context Length: {self.CONTEXT_LENGTH}")
            else:
                logger.warning(f"Total tokens ({max_tokens + self.max_new_tokens}) exceed model context length ({self.CONTEXT_LENGTH}). Reducing max_new_tokens from {self.max_new_tokens} to {new_max_tokens} to fit within context length.")

        output_text = self.inference_chat_model(inputs, max_new_tokens=new_max_tokens)
        
        if return_input_image_size:
            input_height = inputs["image_grid_thw"][0][1]
            input_width = inputs["image_grid_thw"][0][2]

            return output_text, input_height, input_width


        return output_text
