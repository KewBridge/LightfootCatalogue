# Python Modules
import os
import base64
from pathlib import Path
from PIL import Image
import io
# Third-party Modules
from groq import Groq
from lightcat.json_schemas import get_catalogue
# Logging
from lightcat.utils import get_logger
logger = get_logger(__name__)


class GroqBaseModel:

    DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
    MODEL_TYPE = "single"  # Groq models are text-only in this pipeline
    CONTEXT_LENGTH = None
    SUPPORTS_STRICT_SCHEMA = True
    USE_GURANTEED_SCHEMA = True
    SUPPORTS_IMAGES = False

    def __init__(self,
                 batch_size: int = 1,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 api_key: str = None,
                 model_name: str = None):
        """
        Groq model class

        Parameters:
            batch_size (int): batch size for inference
            max_new_tokens (int): Maximum number of tokens
            temperature (float): Model temperature. 0 to 2. Higher the value the more random and
                                 lower the temperature the more focussed and deterministic.
            api_key (str): Optional Groq API key. Defaults to GROQ_API_KEY env var.
            model_name (str): Optional Groq model override.
        """

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        self.client = None


    def load(self):
        """Load the Groq client."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please export GROQ_API_KEY to use Groq models.")

        self.client = Groq(api_key=self.api_key)
        logger.info(f"Groq client initialized for model [{self.model_name}]")


    def unload(self):
        """Unload the Groq client."""
        self.client = None


    def _check(self):
        if self.client is None:
            self.load()

    def _load_images(self, images: list[str]) -> list[str]:
        """
        Load and encode images to data URLs for Groq API.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            List of data URLs (base64-encoded with media type prefix)
        """
        data_urls = []
        
        for img in images:
            if isinstance(img, str):
                # File path
                img_path = Path(img)
                if not img_path.exists():
                    logger.error(f"Image file not found: {img}")
                    continue
                
                # Determine media type
                suffix = img_path.suffix.lower()
                media_type_map = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                media_type = media_type_map.get(suffix, 'image/jpeg')
                
                # Encode to base64 with data URL prefix
                with open(img_path, 'rb') as f:
                    image_data = base64.standard_b64encode(f.read()).decode('utf-8')
                
                data_url = f"data:{media_type};base64,{image_data}"
                data_urls.append(data_url)
                
            elif isinstance(img, Image.Image):
                # PIL Image object
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                image_data = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
                data_url = f"data:image/png;base64,{image_data}"
                data_urls.append(data_url)
            else:
                logger.error(f"Invalid image type: {type(img)}. Must be str or PIL.Image.Image")
        
        return data_urls

    def _flatten_content(self, content, images: list = None) -> list:
        """
        Flatten content into Groq message format, supporting text and images.
        
        Args:
            content: Content to flatten (str, list, dict)
            images: Optional list of data URLs for images
            
        Returns:
            List of content objects for Groq API
        """
        content_parts = []
        
        # Add images first if provided (as image_url objects)
        if images:
            for data_url in images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                })
        
        # Process text content
        text_content = ""
        
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        text_content += c.get("text", "") + " "
                    elif c.get("type") == "image":
                        # Skip image markers, we handle them separately
                        pass
                elif isinstance(c, str):
                    text_content += c + " "
        elif isinstance(content, dict):
            if content.get("type") == "text":
                text_content = content.get("text", "")
        
        # Add text if present
        if text_content.strip():
            content_parts.append({
                "type": "text",
                "text": text_content.strip()
            })
        
        return content_parts if content_parts else [{"type": "text", "text": ""}]

    

    def _normalize_messages(self, conversation, images: list = None) -> list[dict]:
        """
        Normalize conversation to Groq message format.
        
        Args:
            conversation: List of message dicts or strings
            images: Optional list of encoded images
            
        Returns:
            List of normalized messages
        """
        messages = []
        
        for msg in conversation:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    # Flatten content with image support
                    flattened = self._flatten_content(content, images=images if role == "user" else None)
                    messages.append({"role": role, "content": flattened})
                else:
                    messages.append({"role": role, "content": str(content)})
            else:
                raise ValueError("Groq models require messages to be in dict format.")
        
        return messages


    def inference_chat_model(self, messages: list[dict], use_strict: bool = False, pydantic_schema: str = None) -> str:

        params = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

        if use_strict and not self.SUPPORTS_STRICT_SCHEMA:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": pydantic_schema,
                    "strict": self.USE_GURANTEED_SCHEMA,
                    "schema": get_catalogue(pydantic_schema).model_json_schema()
                }
            }


        response = self.client.chat.completions.create(
            **params
        )

        if not response.choices:
            return ""

        return response.choices[0].message.content or ""


    def __call__(self, conversation: list, images: list[str] = None,
                 debug: bool = False, return_input_image_size: bool = False,
                 use_strict: bool = False, pydantic_schema: str = None, **kwargs) -> list:
        """
        Performs inference on the given set of conversations.

        Parameters:
            conversation (list): The input prompt to the model
            images (list): Image paths or PIL Images (supported for vision models)
            debug (bool): Used to print debug prompts
            return_input_image_size (bool): Not supported for Groq models
            use_strict (bool): Whether to use strict schema validation for messages
            pydantic_schema (str): Optional Pydantic schema for message validation

        Return:
            list: A list of model outputs
        """

        if images and not self.SUPPORTS_IMAGES:
            raise ValueError(f"Model {self.model_name} does not support images. Use a vision model.")

        if return_input_image_size:
            raise ValueError("Groq models do not return image sizes.")

        self._check()

        # Load and encode images if provided
        encoded_images = None
        if images and self.SUPPORTS_IMAGES:
            encoded_images = self._load_images(images)
            if debug:
                logger.debug(f"Loaded {len(encoded_images)} images")

        if conversation and isinstance(conversation, list) and len(conversation) > 0 and isinstance(conversation[0], list):
            outputs = []
            for conv in conversation:
                messages = self._normalize_messages(conv, images=encoded_images)
                outputs.append(self.inference_chat_model(messages, use_strict=use_strict, pydantic_schema=pydantic_schema))
            return outputs

        messages = self._normalize_messages(conversation, images=encoded_images)
        return [self.inference_chat_model(messages, use_strict=use_strict, pydantic_schema=pydantic_schema)]



class Groq_GPT_OSS_20B_Model(GroqBaseModel):
    DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
    SUPPORTS_STRICT_SCHEMA = True
    USE_GURANTEED_SCHEMA = True

class Groq_GPT_OSS_120B_Model(GroqBaseModel):
    DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
    SUPPORTS_STRICT_SCHEMA = True
    USE_GURANTEED_SCHEMA = True

class Groq_GPT_OSS_Safeguard_20B_Model(GroqBaseModel):
    DEFAULT_MODEL_NAME = "openai/gpt-oss-safeguard-20b"
    SUPPORTS_STRICT_SCHEMA = True
    USE_GURANTEED_SCHEMA = False

class LLAMA4_17B_Maverick_Model(GroqBaseModel):
    DEFAULT_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
    SUPPORTS_STRICT_SCHEMA = True
    USE_GURANTEED_SCHEMA = False
    SUPPORTS_IMAGES = True

class LLAMA4_17B_Scout_Model(GroqBaseModel):
    DEFAULT_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    SUPPORTS_STRICT_SCHEMA = True
    USE_GURANTEED_SCHEMA = False
    SUPPORTS_IMAGES = True