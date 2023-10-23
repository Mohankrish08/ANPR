# Import necessary libraries
from ultralytics import YOLO
import cv2
import math
import csv
import numpy as np
from sort import *
import cvzone
import easyocr
import datetime
from helper import *

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []
totalCountDown = []
frame_results_list1 = []
limitsUp = [0, 600, 620, 600]
limitsDown = [600, 600, 1500, 600]
frame_number = 0

# coco weights
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load YOLO models
coco_model = YOLO('yolov8n.pt')
model = YOLO('./best.pt')

# Open the video stream
cap = cv2.VideoCapture('./demo.mp4')

plate = []

# Initialize CSV files
entry_csv_file = open('entry.csv', mode='w', newline='')
exit_csv_file = open('exit.csv', mode='w', newline='')
entry_csv_writer = csv.writer(entry_csv_file)
exit_csv_writer = csv.writer(exit_csv_file)
entry_csv_writer.writerow(['License Plate', 'Timestamp'])
exit_csv_writer.writerow(['License Plate', 'Timestamp'])

while True:
    ret, frame = cap.read()
    detections = np.empty((0, 5))

    results = model(frame)[0]
    coco_results = model(frame)[0]

    for r in coco_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.7:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
            cv2.putText(frame, f'{score:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
            currentArray = np.array([x1, y1, x2, y2, score])
            detections = np.vstack((detections, currentArray))

            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Reading license plate
            license_plate = read_license_plate(license_plate_crop)

            # Check the format
            if len(license_plate) == 9 and not license_plate.startswith('X') and license_plate[0].isalpha():
                plate.append(license_plate)
                cv2.putText(frame, str(license_plate), (int(x2 + 10), int(y2 + 10)), cv2.FONT_HERSHEY_DUPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

    resultsTracker = tracker.update(detections)

    cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255, 0, 0), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

                # Store license plate number and timestamp in entry.csv
                if id in plate:
                    entry_csv_writer.writerow([plate[id], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255, 255, 255), 5)

                # Store license plate number and timestamp in exit.csv
                if id in plate:
                    exit_csv_writer.writerow([plate[id], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    frame_results = {
        'plate_unique': list(set(plate)),
        'total_count': totalCount
    }
    
    frame_results_list1.append(frame_results)
    frame_number += 1

    def write_csv(results, entry_csv_path, exit_csv_path):
        timestamp = datetime.datetime.now()
        hour = timestamp.hour

        with open(entry_csv_path if hour < 12 else exit_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['License Plate', 'Timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            for result in results:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for plate in result['plate_unique']:
                    writer.writerow({'License Plate': plate, 'Timestamp': timestamp})

    results = [
    {'plate_unique': ['ABC123', 'XYZ789'], 'total_count': [1, 2]},
    {'plate_unique': ['DEF456'], 'total_count': [3]}
    ]

    entry_csv_path = 'entry.csv'
    exit_csv_path = 'exit.csv'

    write_csv(results, entry_csv_path, exit_csv_path)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close CSV files
entry_csv_file.close()
exit_csv_file.close()

# Print the final counts and unique license plates
print(totalCount)
print(list(set(plate)))
