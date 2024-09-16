import os
import numpy as np
import cv2
import pandas as pd
from flask import Flask, request, render_template
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

        combined_data = []  # To store all the detection results
        room_totals = {}
        knowledge_base_counts = {}

        all_processed_images = []  # To hold all processed image filenames

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

            # Read the image
            image = cv2.imread(file_path)

            # Perform object detection
            result_data, image_with_boxes = detectObjects(image, net, LABELS, COLORS, CONFIDENCE_THRESHOLD, knowledge_base_file)

            # Save the processed image with bounding boxes
            processed_image_path = os.path.join('static', file.filename)
            cv2.imwrite(processed_image_path, image_with_boxes)
            all_processed_images.append(file.filename)  # Append processed image filename

            # Accumulate totals for the room
            for obj_name, total_kb, detected_count in result_data:
                combined_data.append({
                    'Object_Name': obj_name,
                    'Available_Count': total_kb,
                    'Detected_Total_Count': detected_count
                })

        # Combine all the results into a single DataFrame
        if combined_data:
            df = pd.DataFrame(combined_data)
            final_table = df.groupby('Object_Name').sum().reset_index()

            # Append data to CSV file
            append_to_csv(room_number, final_table)

            # Convert DataFrame to HTML table for rendering
            result_table = final_table.to_html(index=False)

        # Render the final processed results
        return render_template('index.html', all_processed_images=all_processed_images, result_table=result_table)

    return render_template('index.html')





if __name__ == "__main__":
    app.run(debug=True)
