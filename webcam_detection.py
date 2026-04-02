import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam (best for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam started... Press ESC to exit")

# Create resizable window
cv2.namedWindow("Webcam Detection", cv2.WINDOW_NORMAL)

# Set fullscreen
cv2.setWindowProperty("Webcam Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()

    # Safety check
    if not ret or frame is None:
        print("Failed to grab frame")
        continue

    # Resize for better performance
    frame = cv2.resize(frame, (1280, 720))

    try:
        # YOLO detection
        results = model(frame)[0]
    except Exception as e:
        print("YOLO Error:", e)
        continue

    # Draw detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Webcam Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()