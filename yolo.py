from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")

results = model.predict(source="./input_images/soccer.jpg", show=True)