# -*- coding: utf-8 -*-

import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import os

def filter_lines(lines : list, middle_line: int, margin_perc: float = 0.20):
    """
    Given a list of possible lines and a presumed middle line, filter out the possible lines that are closer to the middle for inspection

    Parameters:
        lines (list): a list of all possible lines
        middle_line: the x axis that is the presumed middle
        margin_perc: defines how close to the presumed middle line the possible lines should be.
                     smaller the closer to the presumed middle line
    """
    # Calculate margin
    margin = margin_perc * middle_line

    # Get all lines close to the assumed middle_line
    middle_lines = []

    # Quick check for empty list of lines.
    if  lines is None or lines.size == 0:
        return middle_lines
    # Iterate over all lines
    for line in lines:
        # Get the Coords
        x1, y1, x2, y2 = line[0]

        # Find the average x value
        average_x = (x1 + x2) // 2

        # Calculate the angle
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # If the line is a vertical line and if it is closer to the middle line then classify them as the middle line
        if (70 <= angle <= 90) and ( (middle_line - margin) < average_x < (middle_line + margin)):
            middle_lines.append(line[0])

    return middle_lines

def getLines(image: object):
    
    if isinstance(image, str):
        image = cv2.imread(image)

    # Convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Find edges using Canny edge detection
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    # Find lines using Hough Lines Transformation
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=200, minLineLength=80)
    
    return lines

def split_image(path, name = None):

    if isinstance(path, str):
        image = cv2.imread(path)
        name = path
    else:
        image = path
    # Get all lines that correspond with the edges
    lines = getLines(image)
    line_length = len(lines) if lines is not None else 0
    # Find height and width of the image
    height, width = image.shape[:2]
    # Get the middle line X value
    middle_line = width // 2

    # Filter the middle lines
    middle_lines = filter_lines(lines, middle_line)

    if len(middle_lines) > 0:
        # Calculate the line at which to split (calculated as the average middle lines)
        split_line = int(sum([ (x2 + x1)/2 for (x1, _, x2, _) in middle_lines]) / len(middle_lines))

        image_1 = image[:, :split_line]
        image_2 = image[:, split_line:]
        print(f">>>> Splitting successfull : {name} | number of middle lines found = {len(middle_lines)} | number of lines found = {line_length}")
        #Return splitted images
        return image_1, image_2
    else:
        #Return the unsplit image and a null image
        print(f">>>> Splitting unsuccessful  : {name} | number of middle lines found = {len(middle_lines)} | number of lines found = {line_length}")
        print("\tTrying now with thresholding")
        
        return split_image_with_thresholding(path, name)

def split_image_with_gradient(path, name = None):

    if isinstance(path, str):
        image = cv2.imread(path)
        name = path
    else:
        image = path

    h, w, _ = image.shape

    if w < h:
        print("Width of image is less than height")
        print("Considering this image to already be split")
        print(f"Skipping image: {name}")

        return image, None

    # Convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Find edges using Canny edge detection
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Sum the edges to get pixel density at x locale
    edges_sum = np.sum(edges, axis=0)
    # Get the middle region (35% of width as start and 65% of width as end)
    mid_start, mid_end = int(w * 0.4), int(w * 0.6)
    region = edges_sum[mid_start:mid_end]

    # Calculating gradient to detemine location of sharp change (due to a single vertical split between pages)
    gradient = np.gradient(region)

    # Determining the splitting index as the point of largest change
    # Adding the index to the region starting index to bring it back to the full image
    split_index = mid_start + np.argmax(np.abs(gradient))

    image_1 = image[:, :split_index]
    image_2 = image[:, split_index:]

    print(f"Image {name} successfully split down the middle")
    return image_1, image_2

def split_image_with_thresholding(path, name = None):

    if isinstance(path, str):
        image = cv2.imread(path)
        name = path
    else:
        image = path

    h, w, _ = image.shape

    if w < h:
        print("Width of image is less than height")
        print("Considering this image to already be split")
        print(f"Skipping image: {name}")

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
