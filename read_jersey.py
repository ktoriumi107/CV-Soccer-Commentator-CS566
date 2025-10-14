import cv2
from PIL import Image
import pytesseract
import numpy as np
import os

def get_text(path, color_filter):

    img = cv2.imread(path)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # pre-process - https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html

    # get the contours now

    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #img = remove_background(img)

    display_boxes(img, contours)

    if color_filter is not None:
        img = filter_color(img, color_filter)
    
    # convert to PIL Image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #pil_img = Image.fromarray(img)

    # try tesseract - config is very important https://muthu.co/all-tesseract-ocr-options/
    #print("Resulting text for ", path, ": ", pytesseract.image_to_string(pil_img, config="--psm 11"))

def filter_color(img, color_filter):
    # TODO these bounds should be saved in color_filter as touple if incorporating more colors than white
    lower_bound = np.array([200,200,200])
    upper_bound = np.array([255, 255, 255 ])

    mask = cv2.inRange(img, lower_bound, upper_bound)

    img = cv2.bitwise_and(img, img, mask=mask)

    display_image(img)

def display_image(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

def display_boxes(img, contours):
    i = 0

    for contour in contours:
        i += 1

        x,y,w,h = cv2.boundingRect(contour)
        roi=img[y:y+h,x:x+w]
        cv2.imwrite(str(i) + '.jpg', roi)

    cv2.imshow('img',img)
    cv2.waitKey(0)  

def remove_background(img):
    '''Find the contour that matches the outline of the field if applicable.
    Remove the green field and the pixels outside of the field'''

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # find the green parts (this is the field)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([80, 255, 255])

    field = cv2.inRange(img, lower_green, upper_green)

    not_field = cv2.bitwise_not(field)

    # filter out above the field
    contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    perimeter = max(contours)

    field = np.zeros_like(field)

    cv2.drawContours(field, [perimeter], -1, 255, thickness=cv2.FILLED)

    result = cv2.bitwise_and(img, img, mask=not_field)

    #display_image(result)

    return result

img_folder = "test_images"

color_filter = None

for image in os.listdir(img_folder):
    path = os.path.join(img_folder, image)

    get_text(path, color_filter)
