import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils
RTSP_URL = 'rtsp://vmnavas:Zoft@2025@192.168.5.102:554/stream1'
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture("videoplayback.mp4")
fall_detected = False
down_start_time = None
fall_threshold = 100  # sudden drop in Y
still_time_threshold = 1.5  # seconds

prev_head_y = None
prev_time = time.time()
fall_start_time = None

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lms = result.pose_landmarks.landmark
        h, w, _ = img.shape

        head_y = int(lms[mp_pose.PoseLandmark.NOSE].y * h)
        hip_y = int(lms[mp_pose.PoseLandmark.LEFT_HIP].y * h)
        ankle_y = int(lms[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)

        # Head-to-ankle height (proxy for standing)
        body_height = ankle_y - head_y

        # Vertical fall detection (based on velocity)
        current_time = time.time()
        time_diff = current_time - prev_time

        if prev_head_y is not None and time_diff > 0:
            dy = head_y - prev_head_y
            velocity = dy / time_diff

            # condition 1: fast drop
            if velocity > 300:  # pixels per second
                fall_start_time = current_time

        prev_head_y = head_y
        prev_time = current_time

        # condition 2: compressed posture (low height)
        is_compressed = body_height < h * 0.4  # 40% of frame height

        # condition 3: not recovering
        if fall_start_time and is_compressed:
            if down_start_time is None:
                down_start_time = current_time
            elif current_time - down_start_time > still_time_threshold:
                fall_detected = True
        else:
            down_start_time = None

        # Draw result
        if fall_detected:
            cv2.putText(img, "FALL DETECTED", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            cv2.putText(img, "Monitoring...", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Fall Detection", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()