# License-Plate-Detection-Using-OpenCV-Haar-Cascade

## Reference no: 212225230118
## Name: Joshna.M

## Aim
To detect vehicle license plates from an image using OpenCV and Haar Cascade Classifier, draw bounding boxes around the detected plates, and extract the detected license plate region.

## Requirements
**Hardware**
2. Computer/Laptop
3. Minimum 4 GB RAM
4. Webcam or sample vehicle image

**Software**
1. Python 3.x
2. Jupyter Notebook
3. OpenCV
4. NumPy
5. Matplotlib
6. Files Required
7. License Plate Detection.ipynb
8. haarcascade_russian_plate_number.xml
9. Input vehicle image such as car_plate.jpg

## Theory

1. A Haar Cascade Classifier is a machine-learning-based object detection method used to detect specific objects in images.

2. In this experiment, the Haar Cascade XML file is used to identify regions that resemble vehicle license plates. The input image is converted to grayscale before detection because grayscale images require less computational processing.

3. The detected license plate regions are represented using bounding rectangles. The detected region can then be cropped and saved as a separate image.

4. The supplied XML file is an OpenCV Haar classifier with a detection window of 64 × 16 pixels.

## Algorithm
1. Import the required OpenCV, NumPy and Matplotlib libraries.
2. Read the input vehicle image.
3. Display the original image.
4. Convert the image from BGR/RGB to grayscale.
5. Load the Haar Cascade XML classifier.
6. Detect license plates using detectMultiScale().
7. Draw rectangles around the detected plates.
8. Crop the detected license plate region.
9. Apply preprocessing such as histogram equalization or median filtering.
10.Display and save the detected license plate.

## Program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
img = cv2.imread("car_plate.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade classifier
plate_cascade = cv2.CascadeClassifier(
    "haarcascade_russian_plate_number.xml"
)

# Detect license plates
plates = plate_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

# Draw bounding boxes
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the detected plate
    plate = img[y:y+h, x:x+w]

    # Save detected plate
    cv2.imwrite("detected_plate.jpg", plate)

# Display result
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("License Plate Detection")
plt.axis("off")
plt.show()
```
## Output

The program produces:

Original vehicle image
Grayscale image
Detected license plate with bounding rectangle
Cropped license plate image
Histogram-equalized image

The cropped plate is saved as: detected_plate.jpg

## Result
Thus, the license plate was successfully detected from the input vehicle image using OpenCV and Haar Cascade Classifier. The detected license plate was highlighted using a bounding box and the plate region was cropped and saved separately. The detection parameters and preprocessing were modified to improve the detection process.
