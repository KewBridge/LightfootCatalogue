# Standard and third-party modules
import os
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, RIL

# Custom modules – adjust these imports as needed in your project
import lib.config as config


class ImageProcessor:
    def __init__(self,
                 pad: float = 50.0,
                 resize_factor: float = 0.4,
                 remove_area_perc: float = 0.01,
                 middle_margin_perc: float = 0.20):
        """
        Initialize the ImageProcessor with parameters for cropping and splitting.
        
        Parameters:
            pad (float): Padding (in pixels) to add around the detected ROI.
            resize_factor (float): Factor by which to resize the cropped image.
            remove_area_perc (float): Minimum percentage of the image area required for a detected box.
            middle_margin_perc (float): Margin (as a percentage of the image width) used for filtering lines near the middle.
        """
        self.pad = pad
        self.resize_factor = resize_factor
        self.remove_area_perc = remove_area_perc
        self.middle_margin_perc = middle_margin_perc

    def pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        """
        Conver PIL image to cv2 image
        """

        cv2_image = np.array(image)

        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

        return cv2_image

    def cv2_to_pil(self, image: np.ndarray) -> Image.Image:
        """
        Convert cv2 image to PIL image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(image)

    def box_area(self, box: dict) -> int:
        """
        Return the area of of the bounding box
        """
        return box['w'] * box['h']

    # ================= ROI Detection & Cropping Methods =================

    def identify_roi(self, path: str) -> list[int]:
        """
        Identify the region of interest in an image to crop / zoom into

        Parameters:
            path (str): The path to the image
        Return:
            boxes (list): a box (x1, y1, x3, y3) to crop the image
        """
        # Open the image and compute its area.
        image = Image.open(path)
        image_area = image.size[0] * image.size[1]

        x = y = None
        w = h = None

        # Load the Tesseract OCR API and perform detection
        with PyTessBaseAPI(psm=PSM.SINGLE_COLUMN) as api:
            # Start API by setting image and perform recognition
            api.SetImage(image)
            api.Recognize()
            boxes = api.GetComponentImages(RIL.BLOCK, True)

            for i, (im, box, _, _) in enumerate(boxes or []):
                # Skip very small boxes
                if self.box_area(box) < (self.remove_area_perc * image_area):
                    continue

                # im is a PIL image object
                # box is a dict with x, y, w and h keys
                x = box['x'] if x is None else min(x, box['x'])
                y = box['y'] if y is None else min(y, box['y'])
                w = box['x'] + box['w'] if w is None else max(w, box['x'] + box['w'])
                h = box['y'] + box['h'] if h is None else max(h, box['y'] + box['h'])

        return [x, y, w, h]

    def crop_and_resize(self, path: str) -> Image.Image:
        """
        Performs the following task in order:
            1) Identifies the region of interest
            2) Crops the background noise from the image (basically takes only the ROI from the image)
            3) Resizes the image w.r.t aspect ratio
        
        Parameters:
            path (str): Path to the image.
            
        Returns:
            Image: A PIL image that has been cropped and resized.
        """
        # Identify ROI coordinates.
        roi = self.identify_roi(path)
        # Load the image.
        image = Image.open(path)
        # Crop the image (add padding around the ROI).
        cropped = image.crop((
            roi[0] - self.pad,
            roi[1] - self.pad,
            roi[2] + self.pad,
            roi[3] + self.pad
        ))
        w, h = cropped.size

        # Resize the image if the new dimensions (scaled) are reasonably large.
        if (w * self.resize_factor) > 100 and (h * self.resize_factor) > 100:
            cropped = cropped.resize((int(w * self.resize_factor), int(h * self.resize_factor)))

        return cropped

    def crop_image(self, path: str, save_file_name: Optional[str] = None) -> tuple[list[np.ndarray], list[str]]:
        """
        Crop the image, pad it and save the resized image
        
        Parameters:
            path (str): Path to the image.
            save_file_name (str): Format string for the output file name. If None, a default path is used.
            
        Returns:
            tuple: A tuple containing:
                   - A list of split image arrays (cv2/numpy format).
                   - A list of file paths where the images were saved.
        """
        # If no file name is provided, create one based on the original path.
        if save_file_name is None:
            # Build a save directory path.
            save_path = os.path.join(os.sep.join(path.split(os.sep)[:-1]), config.CROPPED_DIR_NAME)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name_parts = path.split(os.sep)[-1].split(".")
            # Create a format string to allow numbering the splits.
            save_file_name = os.path.join(save_path, f"{file_name_parts[0]}_cropped_{{}}.{file_name_parts[-1]}")

        # Crop and resize the image.
        resized = self.crop_and_resize(path)
        # Convert the PIL image to a cv2 image.
        cv2_image = self.pil_to_cv2(resized)
        # Use the split_image method (defined below) to split the image.
        file_display_name = ".".join(path.split(os.sep)[-1].split("."))
        split_imgs = self.split_image(cv2_image, name=file_display_name)

        counter = 0
        saved_paths = []
        for img in split_imgs:
            counter += 1
            if img is not None:
                out_name = save_file_name.format(counter)
                pil_img = self.cv2_to_pil(img)
                pil_img.save(out_name)
                saved_paths.append(out_name)

        return split_imgs, saved_paths

    def __call__(self, images: list[str], save_file_name: Optional[str] = None) -> list[str]:
        """
        Crop the image, pad it and save the resized image

        Parameters:
            images (List[str]): List of image file paths.
            save_file_name (str): Format string for the output file names.
            
        Returns:
            List[str]: A list of saved image file paths.
        """
        new_image_paths = []
        for image_path in images:
            _, paths = self.crop_image(image_path, save_file_name)
            new_image_paths.extend(paths)
        return new_image_paths

    # ================= Splitting Methods =================

    def get_lines(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Given an image, perform binary thresholding, Canny edge detection
        and Hough Lines Transformation to detect all lines in the image.
        
        Parameters:
            image (Union[str, np.ndarray]): The image or path to the image.
            
        Returns:
            np.ndarray: Array of detected lines.
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        # Convert it to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Perform binary thresholding.
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        # Find edges using Canny edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        # Find lines using Hough Lines Transformation
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=200, minLineLength=80)
        return lines

    def filter_lines(self, lines: list, middle_line: int) -> list:
        """
        Given a list of possible lines and a presumed middle line, filter out the possible lines that are closer to the middle for inspection

        Parameters:
            lines (List): List of lines (each line is a 4-tuple: x1, y1, x2, y2).
            middle_line (int): The x-coordinate presumed to be the center.
            
        Returns:
            middle_lines (list): Lines that are nearly vertical and near the middle.
        """
        margin = self.middle_margin_perc * middle_line
        middle_lines = []

        # Quick check for empty list of lines.
        if lines is None or (hasattr(lines, "size") and lines.size == 0):
            return middle_lines

        # Iterate over all lines
        for line in lines:
            # Get the Coords
            x1, y1, x2, y2 = line[0]
            # Find the average x value
            average_x = (x1 + x2) // 2
            # Calculate the angle in degrees.
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # If the line is a vertical line and if it is closer to the middle line then classify them as the middle line
            if (70 <= angle <= 90) and ((middle_line - margin) < average_x < (middle_line + margin)):
                middle_lines.append(line[0])

        return middle_lines

    def split_image(self, image: Union[str, np.ndarray], name: str = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Split the image down the presumed page break using detected vertical lines.
        
        If no suitable lines are found, the method falls back to threshold-based splitting.
        
        Parameters:
            image (Union[str, np.ndarray]): Image (or path) to split.
            name (str): A display name for logging.
            
        Returns:
            Tuple: Two image arrays (left and right halves). If splitting is unsuccessful,
                   the second returned image may be None.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            name = image if name is None else name

        # Get all lines that correspond with the edges
        lines = self.get_lines(image)
        line_length = len(lines) if lines is not None else 0
        # Find height and width of the image
        height, width = image.shape[:2]
        # Get the middle line X value
        middle_line = width // 2
        middle_lines = self.filter_lines(lines, middle_line)

        if len(middle_lines) > 0:
            # Compute the average x-position of the detected middle lines.
            split_line = int(sum([(line[0] + line[2]) / 2 for line in middle_lines]) / len(middle_lines))
            image_left = image[:, :split_line]
            image_right = image[:, split_line:]
            print(f">>>> Splitting successful: {name} | Middle lines found: {len(middle_lines)} | Total lines: {line_length}")
            return image_left, image_right
        else:
            print(f">>>> Splitting unsuccessful: {name} | Middle lines found: 0 | Total lines: {line_length}")
            print("\tFalling back to threshold-based splitting.")
            return self.split_image_with_thresholding(image, name)

    def split_image_with_thresholding(self, image: Union[str, np.ndarray], name: str = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Split the image using thresholding and vertical projection if Hough-based detection fails.
        
        Parameters:
            image (Union[str, np.ndarray]): Image (or path) to split.
            name (str): A display name for logging.
            
        Returns:
            Tuple: Two image arrays (left and right halves). If splitting is unsuccessful,
                   the second returned image may be None.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            name = image if name is None else name

        h, w, _ = image.shape

        # If the image is portrait, assume it is already split.
        if w < h:
            print(f"Width < Height. Skipping splitting for image: {name}")
            return image, None

        # Convert it to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Perform thresholding
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Zoom in on the expected region
        mid_start, mid_end = int(w * 0.4), int(w * 0.6)
        region = binary[:, mid_start:mid_end]

        vert_proj = np.sum(region, axis=0)
        vert_proj = vert_proj / vert_proj.max()
        
        split_start = np.argmin(vert_proj)
        split_end = np.argmin(vert_proj[::-1])

        split_index = int(mid_start + ((split_start + (len(vert_proj) - split_end)) // 2))
        
        image_1 = image[:, :split_index]
        image_2 = image[:, split_index:]

        print(f"\tImage {name} successfully split down the middle")
        return image_1, image_2


