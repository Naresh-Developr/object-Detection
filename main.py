import os
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from src.detect_objects import detectObjects
from src.load_kb import loadKnowledgeBase
from src.to_report import append_to_csv

LABELS_FILE = "weights/coco.names"
CONFIG_FILE = "weights/yolov3.cfg"
WEIGHTS_FILE = "weights/yolov3.weights"
CONFIDENCE_THRESHOLD = 0.3
<<<<<<< HEAD
KNOWLEDGE_BASE_FILE = "KnowledgeBase/101.txt"
CSV_REPORT_FILE = "csv/report.csv"
=======
>>>>>>> origin/sukanth

LABELS = open(LABELS_FILE).read().strip().split("\n")
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)


<<<<<<< HEAD
def drawBoxes(image, layerOutputs, H, W, detected_objects):
    boxes = []
    confidences = []
    classIDs = []
=======
>>>>>>> origin/sukanth

def main():
    st.title("Object Detection with Knowledge Base Comparison")

    uploaded_images = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        room_totals = {}
        knowledge_base_counts = {}

        for uploaded_image in uploaded_images:
            # Extract room number from image file name
            file_name = uploaded_image.name
            room_number = file_name.split("_")[0]  # Extracts the room number before '_'
            st.write(f"Processing image: {file_name} for room {room_number}")

            # Load knowledge base for the corresponding room
            knowledge_base_file = f"KnowledgeBase/{room_number}.txt"
            if not os.path.exists(knowledge_base_file):
                st.error(f"Knowledge base for room {room_number} not found.")
                continue

            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Perform object detection
            result_data, image_with_boxes = detectObjects(image, net, LABELS, COLORS, CONFIDENCE_THRESHOLD, knowledge_base_file)

            # Show the processed image
            st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption=f'Processed Image: {file_name}', use_column_width=True)

            # Accumulate totals for the room
            for obj_name, total_kb, detected_count in result_data:
                if obj_name not in room_totals:
                    room_totals[obj_name] = detected_count
                    knowledge_base_counts[obj_name] = total_kb
                else:
                    room_totals[obj_name] += detected_count

<<<<<<< HEAD
    return image

def detectObjects(image, detected_objects):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    return drawBoxes(image, layerOutputs, H, W, detected_objects)

def updateCsvReport(detected_objects, knowledge_base):
    result_data = []
    for object_name, text_count in knowledge_base.items():
        detected_count = detected_objects.get(object_name, 0)
        result_data.append([object_name, text_count, detected_count])

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(result_data, columns=['Object Name', 'TotalItems (Knowledge Base)', 'DetectedCount'])
    df.to_csv(CSV_REPORT_FILE, index=False)
    return df

# Streamlit app
st.title("Multi-Image Object Detection with Knowledge Base Comparison")

# Multiple image input
uploaded_images = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    detected_objects = defaultdict(int)
    knowledge_base = loadKnowledgeBase(KNOWLEDGE_BASE_FILE)

    for uploaded_image in uploaded_images:
        # Convert uploaded image to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Perform object detection for each image
        image_with_boxes = detectObjects(image, detected_objects)

        # Display the processed image with bounding boxes
        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption=f'Processed Image: {uploaded_image.name}', use_column_width=True)

    # Update CSV report with aggregated detected object counts
    df = updateCsvReport(detected_objects, knowledge_base)

    # Display the comparison between knowledge base and detected objects
    st.write(df)
    st.success(f"Report successfully updated and saved to {CSV_REPORT_FILE}")
=======
        # Final report for the room
        if room_totals:
            # Create DataFrame with object name, available count from knowledge base, and detected total count
            data = {
                'Object_Name': list(room_totals.keys()),
                'Available_Count': [knowledge_base_counts[obj] for obj in room_totals.keys()],
                'Detected_Total_Count': list(room_totals.values())
            }
            df = pd.DataFrame(data)
            st.write(f"Final Report for Room {room_number}")
            st.dataframe(df, use_container_width=True)

            # Append data to CSV file
            append_to_csv(room_number, df)

if __name__ == "__main__":
    main()
>>>>>>> origin/sukanth
