import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model (for person detection)
model = YOLO('yolov8n.pt')  # Use yolov8n.pt for speed or yolov8s.pt for accuracy

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect using YOLOv8
    results = model(frame)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot() 

    cv2.imshow('YOLOv8 Person Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
