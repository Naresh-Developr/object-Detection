import os
import numpy as np
import cv2
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from src.detect_objects import detectObjects
from src.to_report import append_to_csv

app = Flask(__name__)

# YOLO model files
LABELS_FILE = "weights/coco.names"
CONFIG_FILE = "weights/yolov3.cfg"
WEIGHTS_FILE = "weights/yolov3.weights"
CONFIDENCE_THRESHOLD = 0.3

# Load YOLO model and labels
LABELS = open(LABELS_FILE).read().strip().split("\n")
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

# Index route for file upload
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist("images")
        if not files:
            return "No file uploaded!", 400

        room_totals = {}
        knowledge_base_counts = {}
        
        for file in files:
            # Save uploaded file to the uploads folder
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            # Extract room number from file name
            room_number = file.filename.split("_")[0]
            
            # Load the knowledge base file
            knowledge_base_file = f"KnowledgeBase/{room_number}.txt"
            if not os.path.exists(knowledge_base_file):
                return f"Knowledge base for room {room_number} not found.", 404

            # Read image
            image = cv2.imread(file_path)
            
            # Perform object detection
            result_data, image_with_boxes = detectObjects(image, net, LABELS, COLORS, CONFIDENCE_THRESHOLD, knowledge_base_file)

            # # Save the processed image with bounding boxes
            # processed_image_path = os.path.join('static', file.filename)
            # cv2.imwrite(processed_image_path, image_with_boxes)
            
            # After saving the image, pass the correct image path to the template
            processed_image_path = file.filename  # Just pass the filename here
            cv2.imwrite(os.path.join('static', processed_image_path), image_with_boxes)

            # Accumulate totals for the room
            for obj_name, total_kb, detected_count in result_data:
                if obj_name not in room_totals:
                    room_totals[obj_name] = detected_count
                    knowledge_base_counts[obj_name] = total_kb
                else:
                    room_totals[obj_name] += detected_count

        # Create a DataFrame with results
        data = {
            'Object_Name': list(room_totals.keys()),
            'Available_Count': [knowledge_base_counts[obj] for obj in room_totals.keys()],
            'Detected_Total_Count': list(room_totals.values())
        }
        df = pd.DataFrame(data)

        # Save the result to CSV and append to a report
        append_to_csv(room_number, df)

        return render_template('index.html', processed_image=processed_image_path, result_table=df.to_html(index=False))

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)