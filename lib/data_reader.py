# Python Modules

import os
import time

# Custom Modules
import lib.config as config

class DataReader:

    CROPPED_DIR_NAME="cropped"
    ALLOWED_EXT=["jpeg", "png", "jpg", "pdf"]

    def __init__(self,
                 data_path: str,
                 extraction_model: object,
                 crop: bool=False,
                 pad: float=100.0,
                 resize_factor: float=0.4,
                 remove_area_perc: float=0.01
                 save_file_name: str=None
             ):

        """
          Data reader class used to read and extract text from any source of input (image or pdfs)

          The class aims to read the data, extract the text from the images, save them to a text file.
          In case a text file with the extracted text already exists, the extracted data is loaded (without any need to do another inference)

          Parameters:
              data_path (str): the path to the folder containig the data or the the path to the pdf
              extraction_model (object / BaseModel): the model used to extract the text from the source 
              crop (bool): Whether to crop the image or not
              pad (float): Padding value for cropped image
              resize_factor (float): the percentage to which the image should be resized to
              remove_area_perc (float): the percentage that defines the which outlier areas to remove during background noise removal
              save_file_name (str): the name of the save file
        """
        self.data_path = data_path
        self.crop = crop
        self.pad = pad
        self.resize_factor = resize_factor
        self.remove_area_perc = remove_area_perc
        self.save_file_name = save_file_name


        # Loading all files under directory
        self.data_files = self.get_data()        

    def load_files(self, path: str=None) -> list:
        """
        Load (Unravel) files (image or pdfs) given a path to a directory or single image

        If a nested directory is given, all images inside said nexted directories are also gathered.

        Parameter:
            path (str): the absolute path to a directory of images or an image

        Return:
            all_files (list): A list of all the possible images/pdfs in a directory
            
        """

        path = self.data_path if path is None else path
        
        # Return just file if path is to an image or pdf (confirms by checking approved extensions)
        if os.path.isfile(path) and (path.split(".")[-1] in self.ALLOWED_EXT):
            return [self.data_path]
        
        all_files = []

        # Traverse through the directory
        for file in os.listdir(path):
            if file in config.IGNORE_FILE:
                continue
            # Get the file path
            file_path = os.path.join(path, file)

            # Check if it is an image or pdf
            if not(os.path.isdir(file_path)):
                if (file.split(".")[-1] in self.ALLOWED_EXT):
                
                    all_files.append(file_path)
        
        return all_files

    
    def get_data(self):

        
        cropped_dir = os.path.join(self.data_path, self.CROPPED_DIR_NAME)
    
        if self.crop and not(os.path.isdir(cropped_dir)):
            images = sorted(self.load_files())
            print(">>> Cropping Images...")
            images = roi.cropAllImages(images, self.pad, self.resize_factor, 
                  self.remove_area_perc, self.save_file_name)
        elif os.path.isdir(cropped_dir):
            images = self.load_files(cropped_dir)
        else:
            images = self.load_files(image_path)

        # return the images sorted wrt to filename
        return sorted(images)

            
    
    def __call__(self):
        pass
