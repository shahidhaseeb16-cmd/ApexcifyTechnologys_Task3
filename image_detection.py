import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return

    results = model(img)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    #  Get screen size 
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    #  Slightly scale image 
    scale = 1.2   
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    #  Create black background 
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    #  Center position 
    x_offset = (screen_width - new_w) // 2
    y_offset = (screen_height - new_h) // 2

    #  Place image on black background 
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    #  Show result 
    window_name = "Detection Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_image("download.jpg")   # change your image path
