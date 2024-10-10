import re
import os
import json
import lib.config as config


def load_images(path: str) -> list:

    if os.path.isfile(path) and (path.split(".")[-1] in config.IMAGE_EXT):
        return [path]
    
    all_files = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not(os.path.isdir(file_path)):
            if (file.split(".")[-1] in config.IMAGE_EXT):
            
                all_files.append(file_path)
        else:
            files = load_images(file_path)
            all_files.extend(files)
    
    return all_files

def clean_json(text: str) -> str:
    """
    Performs cleaning and normalisation of the input string (of JSON format from AI output)
    and returns a str that is loadadble by json library

    Parameters:
        text (str) : the inital text that is of JSON format
    Return:
        cleaned_text : a cleaned format of the initial text that is passable into JSON
    """
    #Remove starting json word
    cleaned_text = re.sub(r'^json', '', text)
    #Remove any non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
    #Make sure any delimeter or text are in double quotation such that string can be turned into a json file
    cleaned_text = re.sub(r'(?<=\w)"\s+(\[[^\]]+\])', r' \1"', cleaned_text)
    #Remove duplicate quotation marks before : delimiter
    cleaned_text = re.sub(r'(?<=")":', r':', cleaned_text)
    #Normalise any spaces or indentation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text


def verify_json(text: str) -> bool:
    """
    Verifies if the input text is of JSON format

    Parameters:
        text (str) : Initial input text to be verified for JSON format
    Return:
        Verification : True or False depending on if the input text is json format or not
    """

    try:
        json_loaded = json.loads(text)
        return True
    except Exception as e:
        print(f"Non JSON format found in input text")
        print(f"Error: \n {e}")

    return False

def dump_json(text: str, file_name: str = "sample.json", save_path : str = None):
    """
    Dumps the input json-verified text into given file name at defined path

    Parameters:
        text (str) : json-verified text to dump into json file
        file_name (str) : file name to dump the json-text in
        save_path (str) : path to save directory (defined in settings if None)
    """
    #Find the save directory in config if not provided
    save_path = save_path if (save_path is not None) else config.SAVE_PATH

    #Load the json text into json
    json_loaded_text = json.loads(text)

    #Save the file
    with open(os.path.join(save_path, file_name), "w") as json_file:
        json.dump(json_loaded_text, json_file, indent=4)

def save_text(text: str, file_name: str = "sample.txt", save_path : str = None):
    #Find the save directory in config if not provided
    save_path = save_path if (save_path is not None) else config.SAVE_PATH
    
    #Save the file
    with open(os.path.join(save_path, file_name), "w") as file_:
        file_.write(text)
        
def save_json(image: str, text: str, save_path: str = None):

    json_text = text.split("```")[1]
    json_text = clean_json(json_text)

    save_path = save_path if (save_path is not None) else os.path.join(os.sep.join(image.split(os.sep)[:-1]), "text_files")

    if not(os.path.exists(save_path)):
        os.mkdir(save_path)
        
    file_name = image.split(os.sep)[-1].split(".")[0] 
    
    if verify_json(json_text):
        dump_json(json_text, file_name+ ".json", save_path)

    save_text(json_text, file_name+ ".txt", save_path)
def save_jsons(image_json_pairs: list, save_path: str = None):

    for image, text in image_json_pairs:
        save_json(image, text, save_path)
    

        