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

def drawBoxes(image, layerOutputs, H, W, knowledge_base):
    boxes = []
    confidences = []
    classIDs = []
    detected_objects = defaultdict(int)

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

    # Return detected objects and knowledge base data for comparison
    result_data = []
    for object_name, text_count in knowledge_base.items():
        detected_count = detected_objects.get(object_name, 0)
        result_data.append([object_name, text_count, detected_count])

    return result_data, image

def detectObjects(image):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    knowledge_base = loadKnowledgeBase(KNOWLEDGE_BASE_FILE)

    result_data, image_with_boxes = drawBoxes(image, layerOutputs, H, W, knowledge_base)
    
    return result_data, image_with_boxes

# Streamlit app
st.title("Object Detection with Knowledge Base Comparison")

# Image input
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Perform object detection
    result_data, image_with_boxes = detectObjects(image)

    # Display the image with bounding boxes
    st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)

    # Display the comparison between knowledge base and detected objects
    df = pd.DataFrame(result_data, columns=['Object Name', 'TotalItems (Knowledge Base)', 'DetectedCount'])
    st.write(df)
