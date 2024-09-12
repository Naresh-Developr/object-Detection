import numpy as np
import cv2
import pandas as pd
import streamlit as st
from collections import defaultdict

LABELS_FILE = "weights/coco.names"
CONFIG_FILE = "weights/yolov3.cfg"
WEIGHTS_FILE = "weights/yolov3.weights"
CONFIDENCE_THRESHOLD = 0.3
KNOWLEDGE_BASE_FILE = "KnowledgeBase/101.txt"
CSV_REPORT_FILE = "csv/report.csv"

LABELS = open(LABELS_FILE).read().strip().split("\n")
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

def loadKnowledgeBase(filePath):
    knowledge_base = {}
    with open(filePath, 'r') as file:
        for line in file:
            item, count = line.strip().split(',')
            knowledge_base[item.strip()] = int(count.strip())
    return knowledge_base

def drawBoxes(image, layerOutputs, H, W, detected_objects):
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected_objects[LABELS[classIDs[i]]] += 1

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
