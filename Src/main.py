import json
from ultralytics import YOLO
import cv2 as cv
import pytesseract
import easyocr
# import keras_ocr
import re
import requests
from config import api_keys

from validate import validate
from info import info

api_key = api_keys[0]

# pipeline = keras_ocr.pipeline.Pipeline()
reader = easyocr.Reader(['en'])
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

cap = cv.VideoCapture("../Videos/shona2.mp4")

model = YOLO("../Train/runs/detect/train/weights/best.pt")

reg_info_list = []
output_file = "output.txt"

while True:
    success, img = cap.read()
    results = model(source=img, show=False, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            roi = img[y1:y2, x1:x2]

            cv.imshow("roi", roi)
            # cv.imshow("img", img)

            gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

            text = reader.readtext(gray_roi)

            try:
                plate_number = text[0][1]
                if validate(plate_number):
                    with open(output_file, 'a') as file:
                        # Write OCR results to the file
                        file.write(f"{plate_number}\n")

                cv.waitKey(0)
            except Exception:
                print("Plate number not found.")
