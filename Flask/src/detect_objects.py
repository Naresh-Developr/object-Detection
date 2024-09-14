import cv2
from .load_kb import loadKnowledgeBase
from .draw_boxes import drawBoxes

def detectObjects(image, net, LABELS, COLORS, CONFIDENCE_THRESHOLD, KNOWLEDGE_BASE_FILE):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    knowledge_base = loadKnowledgeBase(KNOWLEDGE_BASE_FILE)

    result_data, image_with_boxes = drawBoxes(image, layerOutputs, H, W, LABELS, COLORS, CONFIDENCE_THRESHOLD, knowledge_base)
    
    return result_data, image_with_boxes
