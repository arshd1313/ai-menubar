# app.py


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import yaml
from yaml.loader import SafeLoader
import onnxruntime
import pytesseract 
import json



app = FastAPI()
onnx_file_path = '/home/ahuja/Desktop/menubar-detection/ai-menubar/Model5/weights/best.onnx'
yolo = cv2.dnn.readNetFromONNX(onnx_file_path)
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ort_session = onnxruntime.InferenceSession(onnx_file_path)
# Load YAML
with open('data.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)

labels = data_yaml['names']

# Load YOLO model
yolo = cv2.dnn.readNetFromONNX('/home/ahuja/Desktop/menubar-detection/ai-menubar/Model5/weights/best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# app.py


def extract_text_from_box(image, box):
    x, y, w, h = box

    # Ensure the box coordinates are within the image dimensions
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > image.shape[1]:
        w = image.shape[1] - x
    if y + h > image.shape[0]:
        h = image.shape[0] - y

    # Check if the region is valid
    if w > 0 and h > 0:
        roi = image[y:y+h, x:x+w]

        # Convert the region to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(gray_roi, config='--psm 6')

        return text.strip()
    else:
        return ""



def get_prediction(image):
    # Get YOLO prediction from the image
    row, col, d = image.shape
    
    # Convert image into a square image (array)
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    
    # Get prediction from the square array
    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    yolo.setInput(blob)
    preds = yolo.forward()
    
    return preds, input_image, INPUT_WH_YOLO



def non_maximum_suppression(preds, input_image, INPUT_WH_YOLO, labels):
    detections = preds[0]
    boxes = []
    confidences = []
    classes = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WH_YOLO
    y_factor = image_h / INPUT_WH_YOLO

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]

        if confidence > 0.055:
            class_score = row[5:].max()
            class_id = row[5:].argmax()

            if class_score > 0.055:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5*w) * x_factor)
                top = int((cy - 0.5*h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.0025, 0.0045).flatten()

    return index, boxes_np, confidences_np, classes

def draw_bounding_boxes(image, index, boxes_np, confidences_np, classes, labels):
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = int(confidences_np[ind] * 100)
        class_id = classes[ind]
        class_name = labels[class_id]

        text = f'{class_name}: {bb_conf}%'

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(image, (x, y-30), (x+w, y), (255, 255, 255), -1)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

    return image

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    preds, input_image, INPUT_WH_YOLO = get_prediction(image)
    index, boxes_np, confidences_np, classes = non_maximum_suppression(preds, input_image, INPUT_WH_YOLO, labels)
    result_image = draw_bounding_boxes(image.copy(), index, boxes_np, confidences_np, classes, labels)

    detected_objects = []  # List to store detected objects' information

    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = int(confidences_np[ind] * 100)
        class_id = classes[ind]
        class_name = labels[class_id]

        text = extract_text_from_box(image, boxes_np[ind])

        # Append information to the list
        detected_objects.append({
            "class_name": class_name,
            "confidence": bb_conf,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "extracted_text": text
        })

    _, img_encoded = cv2.imencode('.png', result_image)

    # Convert the detected objects list to a JSON string
    detected_objects_json = json.dumps(detected_objects)

    # Return the detected objects as JSON in the response headers
    headers = {"Detected-Objects": detected_objects_json}
    
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png", headers=headers)