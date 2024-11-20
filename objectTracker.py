import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 1, size=(len(classes), 3))  # Normalize colors to [0, 1]

# Loading video
camera = cv2.VideoCapture("movingCars.avi")

while True:
    _, img = camera.read()
    if img is None:
        break  # Exit the loop if no more frames

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Display using matplotlib
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR image to RGB for displaying with matplotlib
    ax.axis('off')  # Hide the axes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]  # Use normalized color
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=2))
            ax.text(x, y - 5, label, color=color, fontsize=12, backgroundcolor='none')
    plt.draw()
    plt.pause(0.01)  # Pause to update the display

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

camera.release()
plt.close()
