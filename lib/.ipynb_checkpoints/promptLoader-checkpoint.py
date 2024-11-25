import os
import yaml


class PromptLoader:
    
    def __init__(self, filename: str = None, default: str = "./prompts/default.yaml"):
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
                if title.lower() != "image" else 
                [dict(type="image"), dict(type="text", text=message)]
            )
            return dict(role=role, content=contents)

        return dict(role=role, content=message)
            
    
    def get_conversation(self):

        conversation = []

        if "system" in self.yaml_prompt:
            for prompt_title, prompt in self.yaml_prompt["system"].items():
                conversation.append(self.get_prompt(prompt_title, prompt))

        if "user" in self.yaml_prompt:
            for prompt_title, prompt in self.yaml_prompt["user"].items():
                conversation.append(self.get_prompt(prompt_title, prompt, role="user"))

        return conversation
                
            

    