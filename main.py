from fastapi import FastAPI, File, UploadFile
import tempfile
import json
import onnxruntime
import numpy as np
import cv2
from fastapi.responses import FileResponse

app = FastAPI()

# Load the ONNX model
onnx_path = "/home/lenovo/temp-image-detec/Model5/weights/best.onnx"
api_uri = onnxruntime.InferenceSession(onnx_path)

def get_temp_file_path(data, file_extension):
    _, temp_filename = tempfile.mkstemp(suffix=f".{file_extension}")
    with open(temp_filename, 'w') as temp_file:
        json.dump(data, temp_file)  # Adjust this based on your model's output format
    return temp_filename

def run_inference_onnx(image_data):
    # Implement your custom ONNX model inference logic here
    input_data = preprocess_image(image_data)
    input_name = api_uri.get_inputs()[0].name
    input_data = np.expand_dims(input_data, axis=0)  # Assuming batch size is 1

    # Check the expected shape of the input tensor
    expected_shape = api_uri.get_inputs()[0].shape
    print(f"Expected shape of input tensor: {expected_shape}")

    result = api_uri.run(None, {input_name: input_data})
    # Convert NumPy arrays to lists for JSON serialization
    result_as_list = [arr.tolist() for arr in result]
    return result_as_list

def preprocess_image(image_data):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # Resize the image to match the expected input shape
    image = cv2.resize(image, (640, 640)).astype(np.float32)
    # Normalize the image
    image = image / 255.0
    # Transpose the image to match the shape (3, 640, 640)
    image = np.transpose(image, (2, 0, 1))
    return image

def save_image(image_data, output_path):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # Save the image using OpenCV
    cv2.imwrite(output_path, image)

def draw_bounding_box(image, bounding_box):
    # Assuming bounding_box is a NumPy array with shape (4,)
    x, y, w, h = bounding_box
    x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers

    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
    confidence = 0.09  # You need to replace this with the actual confidence score
    text = f'Confidence: {confidence}%'
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)



@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Process the result using your custom ONNX model
    processed_result = run_inference_onnx(contents)
    bounding_box = processed_result[0]  # Assuming the bounding box is the first element in the result

    # Save the processed image
    output_image_path = "output.jpg"
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    draw_bounding_box(image, bounding_box)
    cv2.imwrite(output_image_path, image)

    # Return the processed result as a downloadable file
    return FileResponse(output_image_path, filename="output.jpg", media_type="image/jpeg")

