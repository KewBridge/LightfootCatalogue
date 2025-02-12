import os
import lib.config as config
import numpy as np

from PIL import Image
import cv2


def debugPrint(message: str, debug: bool):
    
    if debug:
        print(message)


def pil_to_cv2(image):
    """
    Conver PIL image to cv2 image
    """

    cv2_image = np.array(image)

    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

    return cv2_image

def cv2_to_pil(image):
    """
    Convert cv2 image to PIL image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image)

def box_area(box: dict) -> int:
    """
    Return the area of of the bounding box
    """
    return box['w'] * box['h']

def load_images(path: str) -> list:
    """
    Load (Unravel) images given a path to a directory or single image

    If a nested directory is given, all images inside said nexted directories are also gathered.

    Parameter:
        path (str): the absolute path to a directory of images or an image

    Return:
        all_files (list): A list of all the possible images in a directory including any that are nested (includes the path to them)
    """
    
    # Return just image if path is to an image (confirms by checking approved extensions)
    if os.path.isfile(path) and (path.split(".")[-1] in config.IMAGE_EXT):
        return [path]
    
    all_files = []

    # Traverse through the directory
    for file in os.listdir(path):
        if file in config.IGNORE_FILE:
            continue
        # Get the file path
        file_path = os.path.join(path, file)

        # Check if it is an image
        if not(os.path.isdir(file_path)):
            if (file.split(".")[-1] in config.IMAGE_EXT):
            
                all_files.append(file_path)
        else:
            # If directory then use recursion to find all images under it
            files = load_images(file_path)
            all_files.extend(files)
    
    return all_files