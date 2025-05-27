import cv2
import time
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Video source - uncomment one to use
#cap = cv2.VideoCapture("gettyimages-585853977-640_adpp.mp4")
#cap = cv2.VideoCapture("gettyimages-486787358-640_adpp.mp4")
cap = cv2.VideoCapture("gettyimages-1325099666-640_adpp.mp4")
# cap = cv2.VideoCapture("gettyimages-1732145742-640_adpp.mp4")
# cap = cv2.VideoCapture(0)  # Webcam

# Parameters
REQUIRED_KPTS = [0, 5, 6, 11, 12, 15, 16]
MIN_KPTS_REQUIRED = 5
HORIZONTAL_AR_THRESHOLD = 0.6
MIN_HORIZ_DURATION = 1.5  # seconds

# Person tracking
person_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    current_time = time.time()
    existing_centers = []

    for pose in results.keypoints:
        keypoints = pose.xy[0].cpu().numpy()
        confs = pose.conf[0].cpu().numpy()

        visible_kpts = sum(1 for idx in REQUIRED_KPTS if keypoints[idx][0] > 0 and confs[idx] > 0.3)
        if visible_kpts < MIN_KPTS_REQUIRED:
            continue

        valid_kpts = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
        x1, y1 = valid_kpts.min(axis=0)
        x2, y2 = valid_kpts.max(axis=0)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1

        if w <= 0 or h <= 0:
            continue
        if w > frame.shape[1] * 0.9 or h > frame.shape[0] * 0.9:
            continue

        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if any(abs(cx - center[0]) < 30 and abs(cy - center[1]) < 30 for cx, cy in existing_centers):
            continue
        existing_centers.append(center)

        aspect_ratio = h / w

        matched_id = None
        for pid, data in person_history.items():
            if abs(data['center'][0] - center[0]) < 50 and abs(data['center'][1] - center[1]) < 50:
                matched_id = pid
                break
        if matched_id is None:
            matched_id = len(person_history) + 1

        history = person_history.get(matched_id, {
            'center': center,
            'aspect_history': []
        })

        if aspect_ratio < HORIZONTAL_AR_THRESHOLD:
            history['aspect_history'].append((current_time, aspect_ratio))
        else:
            history['aspect_history'].clear()

        history['center'] = center

        # Check if horizontal posture is held long enough
        fall_detected = False
        if history['aspect_history']:
            duration = current_time - history['aspect_history'][0][0]
            if duration >= MIN_HORIZ_DURATION:
                fall_detected = True

        person_history[matched_id] = history

        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        label = "Fall" if fall_detected else "Normal"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
