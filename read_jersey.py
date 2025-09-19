import cv2
from PIL import Image
import pytesseract
import numpy as np
import os

def display_text(path):

    img = cv2.imread(path)

    # pre-process - https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html

    # get rid of all pixels other than text color

    # starting color is white
    lower_bound = np.array([200,200,200])
    upper_bound = np.array([255, 255, 255 ])

    mask = cv2.inRange(img, lower_bound, upper_bound)

    img = cv2.bitwise_and(img, img, mask=mask)

    #display_image(img)

    # convert to PIL Image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)

    # try tesseract - config is very important https://muthu.co/all-tesseract-ocr-options/
    print("Resulting text for ", path, ": ", pytesseract.image_to_string(pil_img, config="--psm 11"))


def display_image(img):
    cv2.imshow(img)
    cv2.waitKey(0)

img_folder = "test_images"

for image in os.listdir(img_folder):
    path = os.path.join(img_folder, image)

    display_text(path)
