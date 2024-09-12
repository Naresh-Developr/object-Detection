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

LABELS = open(LABELS_FILE).read().strip().split("\n")
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)



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