from ultralytics import YOLO
import cv2 as cv
import easyocr
from config import api_keys
from Utils.sort import *
from datetime import datetime

from validate import validate

cap = cv.VideoCapture("../Videos/shona2.mp4")
# cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("No source found!!")
    exit()

api_key = api_keys[0]
reader = easyocr.Reader(['en'])
model = YOLO("../Train/runs/detect/train/weights/best.pt")
tracker = Sort()
currentId = 0
currentTime = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
fileName = f"output/vrpr_{currentTime}.csv"
frameSize = 50

while True:
    success, img = cap.read()
    if success:
        img_height, img_width, _ = img.shape
        new_img_height = int(img_height * frameSize / 100)
        new_img_width = int(img_width * frameSize / 100)
        dim = (new_img_width, new_img_height)
        resized_img = cv.resize(img, dim)


        detections = np.empty((0, 4))
        results = model(source=img, show=False, stream=True)
        result_resized = model(source=resized_img, show=False, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                currentArray = np.array([x1, y1, x2, y2])
                detections = np.vstack((detections, currentArray))

        resultTracker = tracker.update(detections)

        for result in resultTracker:
            x1, y1, x2, y2, obj_id = result
            x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)

            roi = img[y1:y2, x1:x2]


            # cv.putText(img, f"{obj_id}", (x1, y1 - 10), 0, 1, (0, 0, 255), 1)

            # cv.imshow("plate", img)
            # cv.waitKey(1)

            gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            text = reader.readtext(gray_roi)

            try:
                plate_number = text[0][1]
                if validate(plate_number) and obj_id > currentId:
                    with open(fileName, 'w') as file:
                        file.write(f"{obj_id}, {plate_number}\n")
                        currentId = obj_id

            except IndexError:
                print("Plate number not found.")

        for result in result_resized:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.imshow("main", resized_img)
                cv.waitKey(1)

    else:
        print("End")
        break

cv.destroyAllWindows()
cap.release()
