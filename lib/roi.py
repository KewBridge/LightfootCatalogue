from PIL import Image
import cv2
from tesserocr import PyTessBaseAPI, PSM, RIL, PT
from lib.utils import box_area, pil_to_cv2, cv2_to_pil
import lib.split_pages as spages
import os

def identifyROI(path: str, remove_area_perc: float = 0.005) -> list:
    """
    Identify the region of interest in an image to crop / zoom into

    Parameters:
        path (str): The path to the image
        remove_area_perc (float): the percentage of the image area defining the threshold for outlier boxes

    Return:
        boxes (list): a box (x1, y1, x3, y3) to crop the image
    """

    # Load the Tesseract OCR API and perform detection
    with PyTessBaseAPI(psm=PSM.SINGLE_COLUMN) as api:
        # Load image and calculate image area
        image = Image.open(path)
        image_area = image.size[0] * image.size[1]

        # Start API by setting image and perform recognition
        api.SetImage(image)
        api.Recognize()

        # Define the (x, y, w, h)
        x = None
        y = None
        w = None
        h = None
        # Get the bounding boxes
        boxes = api.GetComponentImages(RIL.BLOCK, True)

        # Iterate through the different boxes and calculate the minimum (x,y) and maximum (x,y)
        for i, (im, box, _, _) in enumerate(boxes):

            #Define a better way to remove outliers (boxes with areas that are too small)
            if box_area(box) < (remove_area_perc * image_area):
                continue
            # im is a PIL image object
            # box is a dict with x, y, w and h keys
            x = box['x'] if x is None else min(x, box['x'])
            y = box['y'] if y is None else min(y, box['y'])
            w = box['x'] + box['w'] if w is None else max(w, box['x'] + box['w'])
            h = box['y'] + box['h'] if h is None else max(h, box['y'] + box['h'])
    
        return [x,y,w,h]


def cropImage(path: str, pad: float = 50.0, resize_factor: float = 0.4, 
              remove_area_perc: float = 0.005, save_file_name: str = None) -> object:
    """
    Crop the image, pad it and save the resized image

    Parameters:
        path (str): Path to image
        pad (float): padding value
        resize_factor (float): resizing factor. By default 40% of the original image size
        remove_area_perc (float): the percentage of the image area defining the threshold for outlier boxes
        save_file_name (str): the name of the save file
    Returns:
        image (Image): cropped image
    """
    # Identify the region of interest to crop 
    roi = identifyROI(path, remove_area_perc)

    # Load the image
    image = Image.open(path)

    # save_path
    if save_file_name is None:
        # Seperate the path to make a directory to save it in
        save_path = os.path.join(os.sep.join(path.split(os.sep)[:-1]), "cropped")

        # Make sure the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #Get the name of the file and extension
        file_name = path.split(os.sep)[-1].split(".")
        # Define the save_file_name and join it with the save path
        save_file_name = file_name[0] + "_cropped_{}." + file_name[-1]
        save_file_name = os.path.join(save_path, save_file_name)
    else:
        pass

    # Crop the image
    #image = image[int(roi[0]-pad):int(roi[2]+pad), int(roi[1]-pad):int(roi[3]+pad)]
    image = image.crop((roi[0]-pad, roi[1]-pad, roi[2]+pad, roi[3]+pad))
    w, h = image.size

    # Resize the image
    if (w * resize_factor) > 100 and (h * resize_factor) > 100:
        resized = image.resize((int(w * resize_factor), int(h * resize_factor)))
        #resized = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)))
    else:
        resized = image

    #resized.save(save_file_name.format(0))
    
    images = spages.split_image_with_gradient(pil_to_cv2(resized), name = ".".join(file_name))

    counter = 0
    image_paths = []
    for i in images:
        counter += 1
        if i is not None:
            i_save_name = save_file_name.format(counter)
            i = cv2_to_pil(i)
            i.save(i_save_name)
            image_paths.append(i_save_name)
    
    return images, image_paths

def cropAllImages(images: list, pad: float = 50.0, resize_factor: float = 0.4, 
              remove_area_perc: float = 0.005, save_file_name: str = None):
    """
    Crop the image, pad it and save the resized image

    Parameters:
        images (list): list of all images (absolute paths to them)
        pad (float): padding value
        resize_factor (float): resizing factor. By default 40% of the original image size
        remove_area_perc (float): the percentage of the image area defining the threshold for outlier boxes
        save_file_name (str): the name of the save file
        save (bool): save the file or not
    Returns:
        None
    """
    # Create a dummy function with the given parameters
    cropImages = lambda image: cropImage(image, pad, resize_factor, 
              remove_area_perc, save_file_name)
    
    #Pass the path to the dummy function and return a list of new image paths
    new_images = []

    for image in images:
        new_image_list = cropImages(image)
        new_images.extend(new_image_list[1])
        
    return new_images
    

    
    