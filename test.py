import numpy as np
import argparse
import cv2

LABELS_FILE = "weights/coco.names"
CONFIG_FILE = "weights/yolov3.cfg"
WEIGHTS_FILE = "weights/yolov3.weights"
CONFIDENCE_THRESHOLD = 0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = "uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

def drawBoxes (image, layerOutputs, H, W):
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

  # Ensure at least one detection exists
  if len(idxs) > 0:
    for i in idxs.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])

      color = [int(c) for c in COLORS[classIDs[i]]]

      cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
      text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Display the image
  cv2.imshow("output", image)

def detectObjects (imagePath):
  image = cv2.imread(imagePath)
  (H, W) = image.shape[:2]

  ln = net.getLayerNames()
  print(type(ln))
  # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
  net.setInput(blob)
  layerOutputs = net.forward(ln)
  drawBoxes(image, layerOutputs, H, W)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required = True, help = "Path to input file")

  args = vars(ap.parse_args())
  detectObjects(args["image"])

  cv2.waitKey(0)
  cv2.destroyAllWindows()
