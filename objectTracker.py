import cv2
import numpy as np
import time  # Import the time module for measuring execution time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layer names
layer_names = net.getLayerNames()

# Correct indexing for output layers (subtract 1 from indices)
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load video for processing
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("the_new_video_is.avi", fourcc , 25, (852, 480))

# Replace the test.mp4 with your video file
camera = cv2.VideoCapture("movingCars.mp4")

# Start the timer to measure execution time
start_time = time.time()

while True:
    _, img = camera.read()
    if img is None:
        break  # Break if video is over
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filter out weak detections
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Calculate rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # Show the image with the detections
    cv2.imshow("Image", img)

    # Break the loop on pressing 'Esc'
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and destroy all OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Calculate and print the total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")
