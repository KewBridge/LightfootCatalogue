# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def filter_lines(lines : list, middle_line: int, margin_perc: float = 0.1):
  # Calculate margin
  margin = margin_perc * middle_line

  # Get all lines close to the assumed middle_line
  middle_lines = []
  # Iterate over all lines
  for line in lines:
    # Get the Coords
    x1, y1, x2, y2 = line[0]

    # Find the average x value
    average_x = (x1 + x2) // 2

    # Calculate the angle
    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

    # If the line is a vertical line and if it is closer to the middle line then classify them as the middle line
    if (75 <= angle <= 90) and ( (middle_line - margin) < average_x < (middle_line + margin)):
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
  lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=200, minLineLength=100)

  return lines

def split_image(path: str):
  # Load Image
  image = cv2.imread(image_path)

  # Get all lines that correspond with the edges
  lines = getLines(image)


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

    return image_1, image_2
  else:

    return image, image
