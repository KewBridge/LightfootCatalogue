# Logging
from lib.utils import get_logger
logger = get_logger(__name__)
import os
from PIL import Image

def save_to_file(file: str, text: str, mode: str="w") -> None:
    """
    Saves the extracted text to the file
    """
    logger.debug(f"Saving text to file: {file}")
    with open(file, mode) as f:
        f.write(text)
    

def load_from_file(file: str) -> str:
    """
    Load the extracted text from the file

    Returns:
        text: the text read from the file

    """

    text = ""
    logger.debug(f"Loading text from file: {file}")
    with open(file, "r") as f:
        text = f.read()
    
    return text


def get_save_file_name(save_path: str, save_file_name: str) -> str:
    """
    Get the ideal name for the save file. This function checks for any duplicates and adds version numbers
    to the end of given save file names to create unique save file names.
    This ensures no overwriting

    Parameters:
        save_path (str): the path to the directory where the save file will be saved
        save_file_name (str): the input name for the save file as given by user.

    Returns:
        final_save_file_name (str): the finalised save file name
    """

    # Load all files under save path as a hashset
    files = {file for file in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, file))}
    
    id = 0
    final_save_file_name = save_file_name

    # Do a conditional loop to find best name for save file
    while f"{final_save_file_name}.json" in files:

        final_save_file_name = f"{save_file_name}_{id}"
        id += 1

    return final_save_file_name
