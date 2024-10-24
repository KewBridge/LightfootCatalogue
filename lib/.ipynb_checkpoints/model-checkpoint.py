from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import lib.config as config
import lib.utils as utils


class QWEN_Model:

    MODEL_NAME = config.MODEL

    def __init__(self, 
                 prompt: str = None, # Input prompt into the model
                 conversation :list = None, # Input conversation into the model
                 batch_size: int = 3, # Batch size for inference
                 max_new_tokens: int = 5000, # Maximum number of tokens
                 save_path: str = None # Where to save the outputs
                ):

        # Load parameters
        self.prompt = prompt
        self.conversation = self.getConversation(conversation)
        self.batch_size = batch_size if (batch_size is not None) else config.BATCH_SIZE
        self.max_new_tokens = max_new_tokens if (max_new_tokens is not None) else config.MAX_NEW_TOKENS
        self.save_path = save_path

        # Load model
        self.model = self.load_model()
        # Load processor
        self.processor = self.load_processor()

    def getConversation(self, conversation: list) -> list:

        if self.prompt is None and conversation is None:
            return config.CONVERSATION
        elif self.prompt is not None and conversation is None:
            return config.get_conversation(self.prompt)
        else:
            return conversation
            
        
    def load_model(self) -> object:
        """
        Load the Qwen2-VL-7B pretrained model, automatically setting to available device (GPU is given priority if it exists).
    
        Return:
            model (object): Returns the loaded pretrained model.
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
    
        return model


    def load_processor(self) -> object:
        """
        Loads the pre-processor that is used to pre-process the input prompt and images.
    
        Return:
            processor (object): Returns the loaded pretrained processor for the model.
        """
        processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
    
        return processor

    def __call__(self, images: list, debug: bool=False) -> list:
        """
        Performs inference on the given set of images.
    
        Parameters:
            images (list): A set of images to batch inference.
    
        Return:
            output_text (list): A set of model outputs for given set of images.
        """

        # Process the input conversation
        utils.debugPrint("\tProcessing text prompts...", debug)
        text_prompt = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True)
        # Get N text_prompts for equal number of images
        text_prompts = [text_prompt] * len(images)

        utils.debugPrint("\tReading Images...", debug)
        # Open the images from the paths
        images_opened = [Image.open(image) for image in images]
        # Preprocess the inputs
        utils.debugPrint("\tProcessing inputs...", debug)
        inputs = self.processor(
            text=text_prompts, images=images_opened, padding=True, return_tensors="pt"
        )

        utils.debugPrint("\tMoving inputs to gpu...", debug)
        # Move inputs to device (model automatically moves)
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu") # 
    
        # Inference
        utils.debugPrint("\tPerforming inference...", debug)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens) 
        # Increasing the number of new tokens, increases the number of words recognised by the model with trade-off of speed
        # 1024 new tokens was capable of reading upto 70% of the input image (pg132_a.jpeg)
        utils.debugPrint("\tInference Finished", debug)
        output_ids = output_ids.cpu()

        utils.debugPrint("\tSeperating Ids...", debug)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        # Using the preprocessor to decode the numerical values into tokens (words)
        utils.debugPrint("\tDecoding Ids...", debug)
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )


        return output_text

    def batch_infer(self, images: list, save: bool = True, debug: bool=False) -> list:
        """
        Perform batch inference on a list of images given a batch size
    
        Parameters:
            images (list): A set of images to batch inference.
    
        Return:
            image_text_pairs (list): A list of tuples containing (image, output_text) pairs
        """
    
        # Define the value for maximum number of tokens and batch size given they are empty (None)
        print(f">>> Using: \n \tMaximum new tokens = {self.max_new_tokens} \n \tBatch size = {self.batch_size} \n \tsave_path = {self.save_path}")
    
        # Seperate the input images into batches (return the rest as a single batch)
        # Such that given 10 images and a batch size of 3
        # We get 3 batches of size 3 and a final batch of size 1
        batched_images = [images[x:min(x+self.batch_size, len(images))] for x in range(0, len(images), self.batch_size)]

        all_pairs = []
        # Loop through batched images and perform inference
        for ind, batch in enumerate(batched_images):
            print(f">>> Batch {ind+1} starting...")
            output_text = self(batch, debug)
            utils.debugPrint("\tSeperating Outputs...", debug)
            pairs = [(image, output_text[ind]) for ind, image in enumerate(batch)]
            all_pairs.extend(pairs)
            utils.debugPrint("\tOutputs stored!", debug)
            # Saving the output as soon as we get it for each batch, as not to waste memory or to repeat due to interruptions
            if save:
                print(f"\t== Saving Pairs For Batch {ind+1} ==")
                utils.save_jsons(pairs, save_path)
                print(f"\t== Saving Done For Batch {ind+1}  ==")
        

        return all_pairs

    
