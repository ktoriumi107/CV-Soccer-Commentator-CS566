# using this article https://builtin.com/data-science/python-ocr
# "One of the most common Python OCR tools used is the pytesseract library, a wrapper for the Tesseract OCR engine."

# install Tesseract
# install pytesseract

from PIL import Image
import pytesseract
import numpy as np

# USING SIMPLE IMAGE TOOL

## Testing image with computer text
# img1 = np.array(Image.open("./1_python-ocr.webp"))
# text = pytesseract.image_to_string(img1)
# print(text)
# success

## Testing another image with computer text 
# img2 = np.array(Image.open("../../Desktop/docker.png"))
# text2 = pytesseract.image_to_string(img2)
# print(text2)
# success

## Testing a real life image of computer generated text AND human written text
# img3 = np.array(Image.open("./irl_cia.jpeg"))
# text3 = pytesseract.image_to_string(img3)
# print(text3) 
# # fails to recognize human written text

# Testing a real life image of soccer jersey
# img4 = np.array(Image.open("./soccer.jpg"))
# text4 = pytesseract.image_to_string(img4)
# print(text4) 
# # fails to recognize jersey number

# Testing jersey with name and bigger number
# img5 = np.array(Image.open("./soccer2.webp"))
# text5 = pytesseract.image_to_string(img5)
# print(text5) # fails to recognize name and number, again

# REMOVING NOISE IN IMAGE
import cv2

# img6 = cv2.imread('./input_images/soccer2.webp', 0)
# norm_img = np.zeros((img6.shape[0], img6.shape[1]))
# img6 = cv2.normalize(img6, norm_img, 0, 255, cv2.NORM_MINMAX)
# img6 = cv2.threshold(img6, 100, 255, cv2.THRESH_BINARY)[1]
# img6 = cv2.GaussianBlur(img6, (1, 1), 0)
# cv2.imwrite('./output_images/no_noise_soccer.jpeg', img6)
# print("new image saved\n")

# img6 = np.array(Image.open('./output_images/no_noise_soccer.jpeg'))
# text6 = pytesseract.image_to_string(img6)
# print(text6)
# success for noisy computer generated text
# fail for human text
# fail for soccer jersey, although getting some output

# TEXT LOCALIZATION AND DETECTION
# localization is identifying where in the image it is before what finding what it actually says (according to chatgpt)

from pytesseract import Output

img7 = cv2.imread('./output_images/no_noise_soccer.jpeg')
results = pytesseract.image_to_data(img7, output_type=Output.DICT)

for i in range(0, len(results["text"])):
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    text = results["text"][i]
    conf = int(results["conf"][i])
    if conf > 70:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(img7, (x,y), (x+w,y+h), (0,255,0),2)
        cv2.putText(img7, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 2)
cv2.imwrite("./new_soccer.jpeg", img7)