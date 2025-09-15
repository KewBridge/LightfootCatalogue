from typing import Union
import cv2
from natsort import natsorted
import os
import numpy as np
from lib.utils import get_logger
logger = get_logger(__name__)
import tempfile
from pathlib import Path
import shutil
from PIL import Image
from sklearn.cluster import HDBSCAN
import deepdoctection as dd
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
import gc

class LayoutDetector:
    CONFIG_OVERWRITE = [
        "USE_TABLE_SEGMENTATION=False"
    ]


    def __init__(self):

        self.analyser = dd.get_dd_analyzer(config_overwrite=self.CONFIG_OVERWRITE)  # instantiate the built-in analyzer similar to the Hugging Face space demo

    def unload(self):
        del self.analyser
        gc.collect()
        self.analyser = None

    def load(self):
        if self.analyser is None:
            self.analyser = dd.get_dd_analyzer(config_overwrite=self.CONFIG_OVERWRITE)

    def deskew(self, image_array):
        angle = get_angle(image_array)
        return rotate(image_array, angle)

    def process_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Process a single image for OCR extraction.
        This includes converting to grayscale, removing shadows, denoising, binarization, and morphological operations.

        Parameters:
            image (str or np.ndarray): Path to the image or the image itself as a numpy array

        Returns:
            np.ndarray: Processed image ready for OCR extraction
        """
        assert isinstance(image, (str, np.ndarray)), "Image must be a file path or a numpy array."

        if isinstance(image, str):
            image = cv2.imread(image)

        logger.debug("Step 1: Grayscale Conversion")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        logger.debug("Step 2: Gaussian Blur")
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

        logger.debug(f"Step 3: Deskew the image")
        deskewed = self.deskew(blurred)

        logger.debug("Step 4: Remove shadows")
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 41, 10)

        logger.debug("Step 5: Noise Reduction")
        denoised_image = cv2.bilateralFilter(thresh, 9, 75, 75)

        # logger.debug("Step 4: Binarization (black and White) via Otsu's method")
        # binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # logger.debug("Step 5: Deskewing")
        # logger.debug("Skipping deskewing for now")
        # deskewed_image = binarized_image

        # logger.debug("Step 6: Morphological opening (erosion followed by dilation)")
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # opened_image = cv2.morphologyEx(deskewed_image, cv2.MORPH_OPEN, kernel, iterations=1)

        logger.debug("Step 6: Optional dilation to thicken strokes")
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        processed_image = cv2.dilate(denoised_image, kernel2, iterations=1)

        return processed_image

    def process_images(self, images: Union[list[str], list[np.ndarray]], save: bool = False, save_path : str="./resources/images/temp_images") -> np.ndarray:
        """
        Process a list of images for OCR extraction.

        Args:
            images (list[str]): List of image file paths.
            save (bool, optional): If True, processed images will be saved to disk. Defaults to False.

        Returns:
            np.ndarray: Array of processed images.
        """
        save_dir  = Path(save_path)
        if save:
            if save_dir.exists():
                shutil.rmtree(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)


        processed_images = []
        for image in images:
            processed = self.process_image(image)
            processed_images.append(processed)

            if save:
                img = Image.fromarray(processed)
                img_name = image.split(os.sep)[-1]
                img.save(os.path.join(save_dir, img_name))

        return processed_images


    def get_midpoint(self, box: list[float]) -> tuple[float, float]:
        """
        Return the midpoint of the bounding box

        Args:
            box (list[float]): Bounding box

        Returns:
            tuple[float, float]: Midpoint of the bounding box
        """
        x1,y1,x2,y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2
    

    def pad(self, box: list[float], perc: float = 0.02) -> list[float]:
        """
        Add padding to the bounding box

        Args:
            box (list[float]): Bounding Box
            perc (float, optional): Padding percentage. Defaults to 0.02.

        Returns:
            list[float]: Padded Bounding Box
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        x1 -= width * perc
        y1 -= height * perc
        x2 += width * perc
        y2 += height * perc
        return [x1, y1, x2, y2]

    
    def iou(self, box1: list[float], box2: list[float], thresh: float=0.0, eps: float=1e-5) -> list[float]:
        """
        Calculate the intersection over union metric, basically the overlap percentage, between two boxes.

        Args:
            box1 (list[float]): Bounding Box 1
            box2 (list[float]): Bounding Box 2
            thresh (float, optional): Threshold for IoU. Defaults to 0.0.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.

        Returns:
            list[float]: Intersection over Union (IoU) value.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y3 = min(box1[3], box2[3])

        overlap_w = max(0, x2 - x1)
        overlap_h = max(0, y3 - y1)
        overlap_area = overlap_w * overlap_h

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou_val = overlap_area / (box1_area + box2_area - overlap_area + eps)
        return iou_val > thresh


    def merge_boxes(self, box1: list[float], box2: list[float]) -> list[float]:
        """
        Merge two bounding boxes into one bounding box

        Args:
            box1 (list[float]): Bounding box 1
            box2 (list[float]): Bounding box 2

        Returns:
            list[float]: Merged bounding box
        """
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2]



    def merge_columns(self, columns: list[list[float]]) -> list[list[float]]:
        """
        Merge overlapping columns into single columns

        Args:
            columns (list[list[float]]): List of bounding boxes representing columns

        Returns:
            list[list[float]]: List of merged bounding boxes
        """
        columns_sorted = sorted(columns, key=lambda x: x[0])
        merged_columns = []

        for box in columns_sorted:
            if len(merged_columns) == 0:
                merged_columns.append(box)
            else:
                if self.iou(merged_columns[-1], box):
                    merged_columns[-1] = self.merge_boxes(merged_columns[-1], box)
                else:
                    merged_columns.append(box)
        return merged_columns

    def get_boxes_and_midpoints(self, page) -> tuple[list[list[float]], list[tuple[float, float]]]:
        """
        Get the bounding boxes and midpoints of the bounding boxes from a page

        Args:
            page: A page object from deepdoctection

        Returns:
            tuple[list[list[float]], list[tuple[float, float]]]: List of bounding boxes and list of midpoints
        """
        bboxes = []
        midpoints = []

        for layout in page.layouts:
            if layout.text == "":
                continue
            bbox = layout.bbox

            bboxes.append(bbox)
            midpoints.append(self.get_midpoint(bbox))

        return bboxes, midpoints


    def get_groups(self, bboxes: list[list[float]], midpoints: list[tuple[float, float]]) -> dict[int, list[list[float]]]:
        """
        Group the bounding boxes based on their midpoints using HDBSCAN clustering

        Args:
            midpoints (list[tuple[float, float]]): List of midpoints of the bounding boxes

        Returns:
            dict[int, list[list[float]]]: Dictionary of groups of bounding boxes
        """
        if len(midpoints) == 0:
            return {}

        mid_xs = np.array([midpoint[0] for midpoint in midpoints])
        clustering = HDBSCAN(min_samples=min(10, max(len(mid_xs) // 2, 1))).fit(mid_xs.reshape(-1, 1))

        groups = {}

        for idx, i in enumerate(clustering.labels_):
            if i not in groups:
                groups[i] = []
            groups[i].append(bboxes[idx])

        return groups


    def get_columns(self, groups: dict[int, list[tuple[float, float]]]) -> list[np.ndarray]:
        """
        Get the columns from the image based on the bounding boxes

        Args:
            image (np.ndarray): Image array
            boxes (list[list[float]]): List of bounding boxes
            padding (float, optional): Padding percentage. Defaults to 0.02.

        Returns:
            list[np.ndarray]: List of column images
        """
        columns = []
        for group in groups:
            if group == -1:
                continue
            boxes = np.array(groups[group])
            x1 = min(boxes[:, 0])
            y1 = min(boxes[:, 1])
            x2 = max(boxes[:, 2])
            y2 = max(boxes[:, 3])
            columns.append(self.pad([x1, y1, x2, y2]))
        return columns

    
    def sort_and_process_images(self, images: dict[str, list[np.array]]) -> list[np.array]:
        """
        Sort the images based on the file name and process them

        Args:
            images (dict[str, list[np.array]]): Dictionary of file names and list of image arrays

        Returns:
            list[np.array]: List of processed image arrays
        """
        sorted_images = []
        for file_name in natsorted(images.keys()):
            sorted_images.extend(self.process_images(images[file_name], save=False))
        
        return sorted_images


    def __call__(self, image_path: str) -> list[np.array]:
        """
        Pre-process the images, perform layout detection and seperate into columns of text for OCR

        Args:
            images (str): Path to a directory of images or a path to a single image
            save_path (str, optional): Path to save temporary images. Defaults to "./resources/images/temp_images/".

        Returns:
            list[np.array]: List of processed image arrays.
        """
        # Pre-process the image and save them into a temp folder
        #_ = self.process_images(images, save=True, save_path=save_path)

        if os.path.isdir(image_path):
            analysed_folder = self.analyser.analyze(path=image_path)
        elif os.path.isfile(image_path):
            image_bytes = Path(image_path).read_bytes()
            analysed_folder = self.analyser.analyze(image=image_path, bytes=image_bytes)
        else:
            raise FileNotFoundError(f"No images found under {image_path}. Please pass in either a single image or a directory of images, got {image_path} instead")
        analysed_folder.reset_state()
        doc = iter(analysed_folder)

        
        final_images_dict = {}

        for page in doc:
            file_name = page.file_name
            if file_name not in final_images_dict:
                final_images_dict[file_name] = []

            bboxes, midpoints = self.get_boxes_and_midpoints(page)

            groups = self.get_groups(bboxes, midpoints)

            columns = self.get_columns(groups)

            merged_columns = sorted(self.merge_columns(columns), key=lambda x: x[0])
            
            image = Image.open(page.location)
            
            for col in merged_columns:
                cropped_column = image.crop((col[0], col[1], col[2], col[3]))
                final_images_dict[file_name].append(np.array(cropped_column))

        final_images = self.sort_and_process_images(final_images_dict)

        return final_images