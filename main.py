import cv2
from PIL import Image
import pytesseract
import numpy as np
import os

def get_bounding_boxes(img):
    img2 = remove_background(img)

    # grayscale for processing
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert to binary image in order to find contours
    _, bin_image = cv2.threshold(img2, 127, 255, 0)

    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #cv2.imshow('image with bounding boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_background(img):
    '''Remove pixels that are not within the bounds of the 
    field or are not connected to objects on the field'''

    # first goal is to find the edge of the field
    # filter out non green pixels to estimate the image of the field
    lower_limit = np.array([0, 110, 0])
    upper_limit = np.array([150, 255, 150])    

    mask = cv2.inRange(img, lower_limit, upper_limit)

    #field_estimate = cv2.bitwise_and(img, img, mask=mask)

    # determine the edges of the field
    edges = cv2.Cannny(mask,100,200,5)

    cv2.imshow("green only", field_estimate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

img_folder = "test_images"

for image in os.listdir(img_folder):
    path = os.path.join(img_folder, image)

    get_bounding_boxes(cv2.imread(path))