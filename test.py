import cv2
import numpy as np

# Load your image
image = cv2.imread("your_downloaded_image.jpg")  # Replace with the path to your downloaded image

# Your object detection code goes here
# Assuming you have already obtained the bounding boxes and other information
# For example, let's assume you have a bounding box [x, y, w, h]
# and a confidence score conf for the menu bar

# Example bounding box and confidence score
menu_bar_box = [100, 50, 200, 30]  # Format: [x, y, width, height]
menu_bar_confidence = 0.9

# Draw the bounding box and label
x, y, w, h = menu_bar_box
bb_conf = int(menu_bar_confidence * 100)
class_name = "Menu Bar"

text = f'{class_name}: {bb_conf}%'

cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

# Save or display the result
cv2.imwrite("output_image_with_detection.jpg", image)
cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
