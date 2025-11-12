# PART 2 
# helpful functions are findContours(), drawContours()

import cv2

# Read image and convert to gray
img = cv2.imread("./input_images/soccer.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply binary threshold
edges = cv2.Canny(gray_img, 50, 150)

# Find the contours using findContours() and RETR_EXTERNAL mode (less noisy than RETR_TREE mode)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area = cv2.contourArea(c)
    if 500 < area < 5000:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    # x, y, w, h = cv2.boundingRect(c)
    # if w*h > 1500:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                           param1=100, param2=30, minRadius=5, maxRadius=30)

cv2.imwrite("./contours_drawn5.jpg", img)
