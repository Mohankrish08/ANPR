# importing libraries

import streamlit as st
from streamlit_lottie import st_lottie
from ultralytics import YOLO
import cv2
import math
import csv
import numpy as np
from PIL import Image
from sort import *
import easyocr
import datetime
import base64
import requests
import tempfile
from helper import *
from io import BytesIO

# page config

st.set_page_config(
    page_title="ANPR",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


# Loading animations
def loader_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# loading assets
anpr = loader_url('https://lottie.host/08371a31-a5f8-4173-9a1f-152ae121e08f/xEGnXUsHPi.json')
top_image = loader_url('https://lottie.host/df09d055-df33-4008-91a3-b6c66b6a1fc7/UcTU5EvQWp.json')

# functions 
def download_success():
    st.balloons()
    st.success('✅ Download Successful !!')


# for image 
def read_text_and_draw_boxes(image_path):
    try:
        # Initialize the EasyOCR reader
        reader = easyocr.Reader(['en'], gpu=True)

        # Read text from the image
        image = Image.open(image_path)  # Open the image using PIL
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to a format that OpenCV can work with
        results = reader.readtext(image_cv2)

        for detection in results:
            bbox = detection[0]
            text = detection[1]

            # Extract coordinates of the bounding box
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))

            # Draw the bounding box on the image
            cv2.rectangle(image_cv2, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image_cv2, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save the processed image with bounding boxes
        processed_image_path = 'processed.jpg'
        cv2.imwrite(processed_image_path, image_cv2)

        return processed_image_path  # Return the path to the processed image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# csv downloader
    
def get_binary_file_downloader_html(bin_file, file_label):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_label}.csv">Download {file_label}.csv</a>'
    return href


# title  

st.title(' Automatic Number Plate Recognition 🚘🚙')

# loader file

st_lottie(anpr, width=250)

st.markdown("""<b>
Welcome to our Cutting-Edge Automatic Number Plate Recognition System
Our innovative project is dedicated to automating the capture of vehicle number plates at the entrance 
gates of any company. Our system has proven to be an invaluable asset in recording vehicle entry and exit data, 
complete with precise timestamps.
With a commitment to accuracy, efficiency, and security, we provide a seamless solution for enhancing access control 
and monitoring. Join us in revolutionizing the way you manage vehicle access and security. Discover the future of gate access management today</b>""", unsafe_allow_html=True)

st.write('##')

st.markdown("<h3>Select an activity type:</h3>", unsafe_allow_html=True)

# selection 

selected_type = st.radio(
    "",
    ["Upload Image", "Upload video"],
    index=None
)
                     

# driver code

if selected_type == "Upload Image":

    # st.title(' Automatic Number Plate Recognition 🚘🚙')
    #st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image of car's number plate 🚓", type=["png", "jpg", "bmp", "jpeg"])
    
    if st.button("Extract to text") and uploaded_file is not None:
        # Call the 'process_photo' function and get the processed image
        processed_image = read_text_and_draw_boxes(uploaded_file)
        
        if processed_image is not None:
            # Display the processed image
            st.image(processed_image, channels="BGR", use_column_width=True, caption="Processed Image")

# for vidoes

if selected_type == 'Upload video':
    st.title('Upload your video')

    uploaded_file = st.file_uploader("Upload the video of car number plate :car:", type=["mp4"])
    
    if st.button("Extract to CSV") and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        # Open the video stream
        cap = cv2.VideoCapture(video_path)  # Use the local video_path

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

                        if id in plate:
                            entry_csv_writer.writerow([plate[id], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

                if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
                    if totalCountDown.count(id) == 0:
                        totalCountDown.append(id)
                        cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255, 255, 255), 5)

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

        st.markdown("### Download CSV Files")
        st.markdown("Download entry.csv:")
        st.markdown(get_binary_file_downloader_html('entry.csv', 'entry.csv'), unsafe_allow_html=True)
        st.markdown("Download exit.csv:")
        st.markdown(get_binary_file_downloader_html('exit.csv', 'exit.csv'), unsafe_allow_html=True)

st.sidebar.empty()  # Clear the sidebar
