# Logging
from lightcat.utils import get_logger
logger = get_logger(__name__)

# Import libraries
import os


def save_to_file(file: str, text: str, mode: str="w") -> None:
    """
    Saves the text into the file

    Args:
        file (str): file path to save the text
        text (str): text to be saved
        mode (str, optional): file open mode. Defaults to "w".
    """
    logger.debug(f"Saving text to file: {file}")
    with open(file, mode) as f:
        f.write(text)
    

def load_from_file(file: str) -> str:
    """
    Load the extracted text from the file

    Args:
        file (str): file path to load the text from
    Returns:
        str: the text read from the file
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

    Args:
        save_path (str): the path to the directory where the save file will be saved
        save_file_name (str): the input name for the save file as given by user.

    Returns:
        str: the finalised save file name
    """

    # Load all files under save path as a hashset
    base_filename = f"{save_file_name}.json"
    
    # Check if base name is available
    if not os.path.exists(os.path.join(save_path, base_filename)):
        return save_file_name
    
    # Find next available version
    id = 0
    while True:
        versioned_filename = f"{save_file_name}_{id}.json"
        if not os.path.exists(os.path.join(save_path, versioned_filename)):
            return f"{save_file_name}_{id}"
        id += 1