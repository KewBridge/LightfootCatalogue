import os
import yaml


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

    def __getitem__(self, key):
        return self.yaml_prompt[key]
        
    def update_missing(self, custom, default):
        
        for key, value in default.items():

            if isinstance(value, dict) and key in custom:
                self.update_missing(custom[key], default[key])
            elif key not in custom:
                custom[key] = value

    def load(self):
            

        default_file = self.load_yaml(self.default)
        
        if self.filename == self.default or self.filename is None:
            return default_file

        custom_file = self.load_yaml(self.filename)

        if custom_file["inherit_default"]:
            if "default_file" in custom_file.keys() and custom_file["defualt_file"] is not None:
                default_file = self.load_yaml(custom_file["default_file"])
            self.update_missing(custom_file, default_file)

        return custom_file


    def load_yaml(self, filename):
        """
        Load a yaml file given the filename / path to the yaml file

        Parameters:
            filename: the name of the yaml file or the path to the yaml file

        Returns:
            yaml file: returns the read yaml dict
        """

        if filename is None:
            return None
        
        with open(filename, "r") as f:
            return yaml.safe_load(f)
        
    def get_divisions(self):
        return self.yaml_prompt["divisions"]

    def print_prompt(self, prompt=None, indent=""):

        if prompt is None:
            prompt = self.yaml_prompt
            
        for key, value in prompt.items():

            if isinstance(value, dict):
                print(f"=> {key}")
                self.print_prompt(prompt=value, indent= indent+ "  ")
            else:
                print(indent + f"{key}: {value}\n")


    def get_prompt(self, title, prompt, role="system"):

        message = ""
        if title is not None and title != "setup" and role == "system":
            message += f"## {title.upper()} \n"
            
        if isinstance(prompt, dict):
            for key, value in prompt.items():
                message += f"{key.title()}: {str(value)}\n"
        else:
            message += prompt

        if role == "user":
            contents = (
                [dict(type="text", text=message)] 
                if not("image" in title.lower()) else 
                [dict(type="image"), dict(type="text", text=message)]
            )
            return dict(role=role, content=contents)

        return dict(role=role, content=message)
            
    
    def get_conversation(self, extracted_text: str = None):

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
    
    def getImagePrompt(self):

        image_prompt = "Extract the text from both columns in the image, preserving the structure and formatting, ensure no grammatical correction is performed."

        return [dict(role="user", content=[dict(type="image"), 
                dict(type="text", 
                     text=image_prompt)])]
                
            

    def getJsonPrompt(self, json_text, error):
        """
        Define a system prompt including the errorneous json text and the json verificiation error to fix issue

        Parameters:
            json_text: Errorneous json object in string form
            error: the error exception as denoted by verify json function  
        """

        prompt = f"""
            Fix the following JSON prompt given the error as defined below.

            Json prompt:

            {json_text}

            Error:

            {error}

         """

        return [dict(role="system", content=[dict(type="text", text=prompt)])]
     
