import json
from ultralytics import YOLO
import cv2 as cv
import pytesseract
import easyocr
# import keras_ocr
import re
import requests
from config import api_keys

api_key = api_keys[0]

# pipeline = keras_ocr.pipeline.Pipeline()
reader = easyocr.Reader(['en'])
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

cap = cv.VideoCapture("../Videos/shona2.mp4")

model = YOLO("../Train/runs/detect/train/weights/best.pt")

reg_info_list = []
output_file = "output.txt"


def validate(plate_number):
    pattern = r"^[A-Za-z]{2}[ -]?[0-9]{2}[ -]?[A-Za-z]{1,2}[ -]?[0-9]{4}$"
    match = re.match(pattern, plate_number)
    if match:
        print("Valid plate number:", plate_number)
        # reg_info = info(plate_number, api_key=api_key)
        # print(reg_info)

        with open(output_file, 'a') as file:
            # Write OCR results to the file
            file.write(f"{plate_number}\n")

    else:
        print("Invalid plate number:", plate_number)


def info(reg_no, api_key):
    url = "https://rto-vehicle-information-verification-india.p.rapidapi.com/api/v1/rc/vehicleinfo"

    payload = {
        "reg_no": reg_no,
        "consent": "Y",
        "consent_text": ("I hear by declare my consent agreement "
                         "for fetching my information via AITAN Labs API")
    }

    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": ("rto-vehicle-information-"
                            "verification-india.p.rapidapi.com")
    }

    response = requests.post(url, json=payload, headers=headers)

    return response.json()


def detect():
    while True:
        success, img = cap.read()
        results = model(source=img, show=True, stream=True)
        for result in results:
            boxes = result.boxes
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                roi = img[y1:y2, x1:x2]

                cv.imshow("roi", roi)
                # cv.imshow("img", img)

                gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

                # text = pytesseract.image_to_string(gray_roi)
                # prediction_groups = pipeline.recognize([roi])

                text = reader.readtext(gray_roi)

                try:
                    plate_number = text[0][1]
                    validate(plate_number)

                    cv.waitKey(0)
                except Exception:
                    print("Plate number not found.")


detect()

