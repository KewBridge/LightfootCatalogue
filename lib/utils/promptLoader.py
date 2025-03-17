import os
import yaml
from typing import Optional, Union

class PromptLoader:
    
    def __init__(
            self, 
            filename: str = None, 
            default: str = "./prompts/default.yaml"
            ):
        """
        Prompt Loader class acts as an intermediary between the user prompt and the model.

        Loads the users prompts and returns step-based instructions/prompts to the model when requested

        Parameters:
            filename (str): the file name of the prompt
        """
        self.filename = filename
        self.default = default
        
        self.yaml_prompt = self.load()

    def __getitem__(self, key: str) -> Union[int, str, list]:
        """
            Get value at key

        Args:
            key (str): Key in prompt

        Returns:
            Union[int, str, list]: Value at key
        """
        return self.yaml_prompt[key]
    

    def update_missing(self, custom: dict, default: dict) -> None:
        """
        Update any missing keys in custom from default

        Args:
            custom (dict): custom prompt
            default (dict): default prompt
        """
        
        for key, value in default.items():

            if isinstance(value, dict) and key in custom:
                self.update_missing(custom[key], default[key])
            elif key not in custom:
                custom[key] = value


    def load(self) -> dict:
        """
        Load the prompt

        Returns:
            custom_file (dict): loaded prompt
        """
            

        default_file = self.load_yaml(self.default)
        
        if self.filename == self.default or self.filename is None:
            return default_file

        custom_file = self.load_yaml(self.filename)

        if custom_file["inherit_default"]:
            if "default_file" in custom_file.keys() and custom_file["defualt_file"] is not None:
                default_file = self.load_yaml(custom_file["default_file"])
            self.update_missing(custom_file, default_file)

        return custom_file


    def load_yaml(self, filename: str) -> dict:
        """
        Load a yaml file given the filename / path to the yaml file

        Parameters:
            filename (str): the name of the yaml file or the path to the yaml file

        Returns:
            yaml file (dict): returns the read yaml dict
        """

        if filename is None:
            return None
        
        with open(filename, "r") as f:
            return yaml.safe_load(f)
    

    def get_divisions(self) -> Union[list, str]:
        """
        Return the divisions

        Returns:
            Union[list, str]: Either a list of all division names, or a single division
        """
        
        return self.yaml_prompt["divisions"]

    def print_prompt(self, prompt: dict=None, indent: str="") -> None:
        """
        Print the prompt

        Args:
            prompt (dict, optional): Prompt to print. Defaults to None.
            indent (str, optional): intendation for values. Defaults to "".
        """

        if prompt is None:
            prompt = self.yaml_prompt
            
        for key, value in prompt.items():

            if isinstance(value, dict):
                print(f"=> {key}")
                self.print_prompt(prompt=value, indent= indent+ "  ")
            else:
                print(indent + f"{key}: {value}\n")

    def _unravel_prompt(self, prompt: Union[dict, list]) -> str:
        """
        Unravel the prompt from dict or list into prompt message

        Args:
            prompt (Union[dict, list]): Input prompt

        Returns:
            str: prompt message
        """
        message = ""

        if isinstance(prompt, dict):
            for key, value in prompt.items():
                message += f"{key.title()}: {str(value)}\n"
        elif isinstance(prompt, list):
            for item in prompt:
                message += self._unravel_prompt(item)
        else:
            message += prompt
        
        return message

    def get_prompt(self, title: str, prompt: dict, role="system") -> dict:
        """
        Generate the prompt to the model

        Args:
            title (str): title of the prompt. Key in prompt file
            prompt (dict): Input prompt dict
            role (str, optional): The role to add prompt under. Defaults to "system".

        Returns:
            dict: Return prompt dict to model
        """

        message = ""
        if title is not None and title != "setup" and role == "system":
            message += f"## {title.upper()} \n"
            
        message += self._unravel_prompt(prompt)

        if role == "user":

            if not("\{extracted_text\}" in message):
                message += "\{extracted_text\}"
            contents = (
                [dict(type="text", text=message)] 
                if not("image" in title.lower()) else 
                [dict(type="image"), dict(type="text", text=message)]
            )
            return dict(role=role, content=contents)

        return dict(role=role, content=message)
            
    
    def get_conversation(self, extracted_text: str = None) -> list:
        """
        The input conversation the model

        Args:
            extracted_text (str, optional): Extracted text to be inputted into user prompt. Defaults to None.

        Returns:
            list: Input conversation to model
        """
        conversation = []

        if "system" in self.yaml_prompt:
            for prompt_title, prompt in self.yaml_prompt["system"].items():
                conversation.append(self.get_prompt(prompt_title, prompt))

        #TODO: Need to append this for user input as the extracted text
        if "user" in self.yaml_prompt:
            for prompt_title, prompt in self.yaml_prompt["user"].items():
                if not(extracted_text is None):
                    prompt = prompt.format(extracted_text=extracted_text)
                conversation.append(self.get_prompt(prompt_title, prompt, role="user"))

        return conversation
    
    def getImagePrompt(self) -> list:
        """
        Get image prompt to model

        Returns:
            list: Image prompt to model
        """

        image_prompt = """
                    Extract only the main body text from the image, preserving the original structure and formatting. 
                    Do not perform any grammatical corrections.  
                    Ignore any text found in the top or bottom margins of the image, including headers, footers, page numbers, and page titles. 
                    Only process text located within the central column regions.
                    Text in the top 5\% and bottom 5\% should be ignored.
                    Do not add, invent, or repeat text on your own. If a line appears once in the image, it must appear once in the output. If the same line appears multiple times in the image, reproduce it exactly as many times as it actually appears.
                    """

        return [dict(role="user", content=[dict(type="image"), 
                dict(type="text", 
                     text=image_prompt)])]
                
            

    def getJsonPrompt(self, json_text: str) -> list:
        """
        Define a system prompt including the errorneous json text and the json verificiation error to fix issue

        Parameters:
            json_text (str): Errorneous json object in string form
        
        Returns:
            list: system prompt to fix json error
        """

        prompt = f"""
            Fix the following JSON prompt.

            Json prompt:

            {json_text}

         """

        return [dict(role="system", content=[dict(type="text", text=prompt)])]
     
