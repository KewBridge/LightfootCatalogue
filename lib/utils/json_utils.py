import os
import re
import json
import pandas as pd


def clean_json(text: str) -> str:
    """
    Performs cleaning and normalisation of the input string (of JSON format from AI output)
    and returns a str that is loadadble by json library

    Parameters:
        text (str) : the inital text that is of JSON format
    Return:
        cleaned_text : a cleaned format of the initial text that is passable into JSON
    """
    # Remove starting json word
    cleaned_text = re.sub(r'^json', '', text)
    # Remove any non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
    # Make sure any delimeter or text are in double quotation such that string can be turned into a json file
    cleaned_text = re.sub(r'(?<=\w)"\s+(\[[^\]]+\])', r' \1"', cleaned_text)
    # Remove duplicate quotation marks before : delimiter
    cleaned_text = re.sub(r'(?<=")":', r':', cleaned_text)
    # Normalise any spaces or indentation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text


def verify_json(text: str, clean: bool = False, out: bool = False) -> bool:
    """
    Verifies if the input text is of JSON format

    Parameters:
        text (str) : Initial input text to be verified for JSON format
    Return:
        Verification : True or False depending on if the input text is json format or not
    """

    try:
        text = clean_json(text) if clean else text
        json_loaded = json.loads(text)
        if out:
            return True, json_loaded
        return True
    except Exception as e:
        print(f">>> Non JSON format found in input text")
        print(f">>> Error: \n {e}")

    if out:
        return False, None
    return False

    
def dump_json(text: str, file_name: str = "sample.json", save_path : str = None):
    """
    Dumps the input json-verified text into given file name at defined path

    Parameters:
        text (str) : json-verified text to dump into json file
        file_name (str) : file name to dump the json-text in
        save_path (str) : path to save directory (defined in settings if None)
    """
    # Find the save directory in config if not provided
    save_path = save_path if (save_path is not None) else config.SAVE_PATH

    # Load the json text into json
    json_loaded_text = json.loads(text)

    # Save the file
    with open(os.path.join(save_path, file_name), "w") as json_file:
        json.dump(json_loaded_text, json_file, indent=4)


def save_csv(text: str, file_name: str = "sample.csv", save_path: str = None):
    
    # Find the save directory in config if not provided
    save_path = save_path if (save_path is not None) else config.SAVE_PATH

    # Load the json text into json
    json_loaded_text = json.loads(text)

    tabular = json_to_csv(json_loaded_text)

    tabular.to_csv(os.path.join(save_path, file_name), index=False)


def save_text(text: str, file_name: str = "sample.txt", save_path : str = None):
    """
    Save the given text in a text document for debugging purposes

    Parameters:
        text (str) : text to dump into txt file
        file_name (str) : file name to dump the text in
        save_path (str) : path to save directory (defined in settings if None)
    """
    # Find the save directory in config if not provided
    save_path = save_path if (save_path is not None) else config.SAVE_PATH
    
    # Save the file
    with open(os.path.join(save_path, file_name), "w") as txt_file:
        txt_file.write(text)


def save_json(image: str, text: str, save_path: str = None):
    """
    Perform cleaning and save the input text into a JSON format (if possible) else a Text Document

    Parameters:
        image (str): the path to the image
        text (str): the output text from the model
        save_path (str): the path to save the model in
    """
    print(f">>> Saving text output for {image}")
    # Cleaning the text here
    if "```" in text:
        text = text.split("```")[1]
        
    json_text = clean_json(text)

    # defining the save path if it does not exist
    save_path = save_path if (save_path is not None) else os.path.join(os.sep.join(image.split(os.sep)[:-1]), "text_files")

    # Makes sure th save path exists
    if not(os.path.exists(save_path)):
        os.mkdir(save_path)

    # Get the file name of the image
    file_name = image.split(os.sep)[-1].split(".")[0] 

    # verify if text is json formatted
    if verify_json(json_text):
        # Dump into json file
        print(">>>> Saving JSON...")
        dump_json(json_text, file_name+ ".json", save_path)
        # Dump into csv
        #print(">>>> Saving CSV...")
        #save_csv(json_text, file_name+ ".csv", save_path)
    else:
        # If not, save it as a text file
        save_text(json_text, file_name+ ".txt", save_path)


def save_jsons(image_json_pairs: list, save_path: str = None):
    """
    Iterate thrrough all image-text pairs and save them

    Parameters:
        image_json_pairs (list): image-json pairs to save
        save_path (str): the path in which to save
    """
    for image, text in image_json_pairs:
        
        save_json(image, text, save_path)


def json_to_csv(json_file: dict) -> pd.DataFrame:
    
    tabular = pd.DataFrame(columns=["family", "species", "folder", "contents"])

    try:
        for family, f_val in json_file.items():
            try:
                for species, s_val in f_val.items():
                    try:
                        for folder_name in s_val['folder_contents']:
                            if isinstance(folder_name, str):
                                tabular.loc[len(tabular)] = [family, species, folder_name, None]
                            else:
                                if "contents" in folder_name:
                                    for content in folder_name['contents']:
                                        tabular.loc[len(tabular)] = [family, species, folder_name['folder'], content]
                                else:
                                    tabular.loc[len(tabular)] = [family, species, folder_name['folder'], None]
                
                            if "contents" in s_val and tabular.loc[len(tabular)-1]["contents"] is None:
                                tabular.loc[len(tabular)-1, "contents"] = s_val["contents"][0]
                    except:
                        tabular.loc[len(tabular)] = [family, species, None, None]
            except:
                tabular.loc[len(tabular)] = [family, None, None, None]
    except:
        tabular.loc[len(tabular)] = [None, None, None, None]

    return tabular