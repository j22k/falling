import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Constants and Thresholds ---
# For YOLOv8 Pose Model
MODEL_PATH = 'yolov8n-pose.pt' # Or use yolov8m-pose.pt, yolov8l-pose.pt for potentially better accuracy but slower
MIN_CONF_DETECTION = 0.5 # Minimum confidence for a person detection
MIN_CONF_KPT = 0.3       # Minimum confidence for a keypoint to be considered visible

# Keypoints for Bounding Box calculation (YOLO indices: Nose=0, Shoulders=5,6, Hips=11,12, Ankles=15,16)
# Indices: [0 (Nose), 5 (LShoulder), 6 (RShoulder), 11 (LHip), 12 (RHip), 15 (LAnkle), 16 (RAnkle)]
REQUIRED_KPTS_FOR_BOX = [0, 5, 6, 11, 12, 15, 16]
MIN_KPTS_FOR_BOX = 5 # Minimum number of required keypoints visible for a valid person box

# Keypoints for Angle calculation (YOLO indices: Left Shoulder=5, Left Hip=11, Left Knee=13)
# You could add logic for the right side (6, 12, 14) as well, e.g., take the minimum angle or check both.
# For simplicity, we'll start with the left side as in the MediaPipe example.
ANGLE_KPTS_INDICES = {
    'shoulder': 5, # Left Shoulder
    'hip': 11,     # Left Hip
    'knee': 13     # Left Knee
}
MIN_CONF_FOR_ANGLE_KPTS = 0.3 # Minimum confidence for the specific angle keypoints

# Fall Detection Criteria Thresholds
HORIZONTAL_AR_THRESHOLD = 0.6     # Ratio of height / width
ANGLE_THRESHOLD = 151.03052520589148             # Degrees for the hip angle (e.g., < 60 degrees is considered a collapsed torso)
MIN_FALL_POSE_DURATION = 1.0      # Seconds a potential fall pose must be held

# Tracking Parameters
TRACKING_DISTANCE_THRESHOLD = 150 # Max pixel distance for matching a person across frames
SINGLE_FRAME_NMS_THRESHOLD = 40   # Max pixel distance to consider detections in *same* frame as duplicates (optional cleanup)
HISTORY_CLEANUP_INTERVAL = 5      # Seconds to keep track of a person after they disappear

calculated_angles = []

# Video Source
# cap = cv2.VideoCapture("gettyimages-585853977-640_adpp.mp4")
# cap = cv2.VideoCapture("gettyimages-486787358-640_adpp.mp4")
# cap = cv2.VideoCapture("gettyimages-1325099666-640_adpp.mp4")
# cap = cv2.VideoCapture("gettyimages-1732145742-640_adpp.mp4")
cap = cv2.VideoCapture("videoplayback.mp4")
# cap = cv2.VideoCapture("bowling.mp4") # Example from MediaPipe script
# cap = cv2.VideoCapture(0)  # Webcam

# RTSP_URL = 'rtsp://vmnavas:Zoft@2025@192.168.5.102:554/stream2'
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)

# --- Helper Function ---
def calculate_angle(a, b, c):
    """Calculates the angle in degrees between three points."""
    a = np.array(a) # Shoulder
    b = np.array(b) # Hip (vertex)
    c = np.array(c) # Knee

    # Ensure points are distinct to avoid errors
    if np.array_equal(a,b) or np.array_equal(b,c):
         return 180.0 # Return upright angle if points are the same

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians)) # Convert to degrees
    print(f"Angle calculation: a={a}, b={b}, c={c}, angle={angle:.2f}") # Debug
    # Normalize angle to be between 0 and 180
    if angle > 180.0:
        angle = 360 - angle

    return angle

# --- Load Model ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure you have 'ultralytics' installed (`pip install ultralytics`)")
    print(f"and the model file '{MODEL_PATH}' exists or is accessible.")
    exit()


# --- Person Tracking State ---
# Stores state for each tracked person
person_history = {}
next_person_id = 1
last_cleanup_time = time.time()

# --- Main Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video

    current_time = time.time()

    # Run YOLO inference
    # Setting verbose=False reduces console output
    # conf=MIN_CONF_DETECTION filters initial person detection confidence
    results = model(frame, conf=MIN_CONF_DETECTION, verbose=False)[0]

    # Store detections processed in the current frame to handle potential duplicates
    current_frame_detections_centers = []

    # --- Process Each Detected Person ---
    # results.keypoints gives a list of Keypoints objects, one for each person detected
    for i, pose in enumerate(results.keypoints):

        # --- Check for missing keypoint confidence data (Fix for TypeError) ---
        # Sometimes pose.conf can be None if no keypoints were found for the detection
        if pose.conf is None or len(pose.conf) == 0:
             # print(f"Skipping detection {i}: No keypoint confidences available.") # Debug
             continue # Skip this pose if confidence data is missing


        # Pose keypoints are in pixel coordinates (xy) and their confidences (conf)
        keypoints = pose.xy[0].cpu().numpy() # Shape (17, 2)
        confs = pose.conf[0].cpu().numpy() # Shape (17,)

        # 1. Calculate Bounding Box from Keypoints and Filter based on visible keypoints
        visible_relevant_kpts_coords = []
        for idx in REQUIRED_KPTS_FOR_BOX:
            # Check if index is valid for the keypoints array and keypoint is visible and confident
            if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0 and confs[idx] > MIN_CONF_KPT:
                visible_relevant_kpts_coords.append(keypoints[idx])

        if len(visible_relevant_kpts_coords) < MIN_KPTS_FOR_BOX:
            # print(f"Skipping detection {i}: Not enough visible keypoints ({len(visible_relevant_kpts_coords)}/{MIN_KPTS_FOR_BOX})") # Debug
            continue # Skip this detection if not enough reliable keypoints

        visible_relevant_kpts_coords = np.array(visible_relevant_kpts_coords)
        # Handle potential edge case where min/max might fail on specific numpy array shapes if something is wrong
        try:
            x1, y1 = visible_relevant_kpts_coords.min(axis=0)
            x2, y2 = visible_relevant_kpts_coords.max(axis=0)
        except ValueError:
            # print(f"Skipping detection {i}: Could not calculate min/max for bounding box.") # Debug
            continue # Skip if min/max calculation fails (e.g., empty array somehow)

        # Ensure coords are integers for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            # print(f"Skipping detection {i}: Invalid bounding box dimensions ({w}x{h})") # Debug
            continue # Skip invalid boxes

        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Optional: Simple NMS for current frame detections (handle overlapping boxes for same person by YOLO)
        is_duplicate_in_frame = False
        for existing_center in current_frame_detections_centers:
             dist = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
             if dist < SINGLE_FRAME_NMS_THRESHOLD:
                 is_duplicate_in_frame = True
                 break
        if is_duplicate_in_frame:
             # print(f"Skipping detection {i}: Duplicate in current frame") # Debug
             continue # Skip this pose if it's too close to one already processed in this frame

        current_frame_detections_centers.append(center) # Add this valid detection's center

        # 2. Calculate Aspect Ratio (Height / Width)
        aspect_ratio = h / w

        # 3. Calculate Angle (e.g., Left Shoulder-Left Hip-Left Knee)
        # Check if the required keypoints for the angle calculation are visible and confident
        angle = None # Default angle to None if keypoints are missing or calculation fails
        angle_kpts_coords = {}
        angle_kpts_available = True
        for name, idx in ANGLE_KPTS_INDICES.items():
             # Check if index is valid and keypoint is visible and confident
             if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0 and confs[idx] > MIN_CONF_FOR_ANGLE_KPTS:
                 angle_kpts_coords[name] = (keypoints[idx][0], keypoints[idx][1]) # Use float coords for angle calculation precision
             else:
                 # print(f"Detection {i}: Missing or low confidence keypoint for angle: {name} (idx {idx})") # Debug
                 angle_kpts_available = False
                 break # Need all three points for the angle

        if angle_kpts_available:
             try:
                  # Pass shoulder, hip, knee coordinates to the angle function
                  angle = calculate_angle(angle_kpts_coords['shoulder'], angle_kpts_coords['hip'], angle_kpts_coords['knee'])
                  # print(f"Detection {i}: Calculated angle = {angle:.2f}") # Debug)
             except Exception as e:
                  print(f"Detection {i}: Error calculating angle: {e}")
                  angle = None # Set angle to None if calculation fails
        if angle_kpts_available:
            try:
                # Pass shoulder, hip, knee coordinates to the angle function
                angle = calculate_angle(angle_kpts_coords['shoulder'], angle_kpts_coords['hip'], angle_kpts_coords['knee'])
                # Store the calculated angle
                calculated_angles.append(angle)
                # print(f"Detection {i}: Calculated angle = {angle:.2f}") # Debug)
            except Exception as e:
                print(f"Detection {i}: Error calculating angle: {e}")
                angle = None 

        # 4. Match with Person History (Simple Tracking)
        matched_id = None
        # Iterate through existing persons to find a match based on proximity
        # Use list(person_history.items()) to iterate over a copy in case we add a new ID during iteration
        for pid, data in list(person_history.items()):
             # Calculate distance from the historical center
             dist = np.sqrt((center[0] - data['center'][0])**2 + (center[1] - data['center'][1])**2)
             if dist < TRACKING_DISTANCE_THRESHOLD:
                 matched_id = pid
                 break

        if matched_id is None:
             # This is a new person entering the frame
             matched_id = next_person_id
             next_person_id += 1
             person_history[matched_id] = {
                 'center': center,
                 'is_potential_fall_pose': False, # True if current pose is horizontal or low angle
                 'fall_pose_start_time': None,     # Time when potential fall pose started
                 'fall_detected': False,           # True if duration threshold is met
                 'last_seen': current_time         # Time when this person was last seen
             }
             # print(f"New person detected: ID {matched_id}") # Debug)
        else:
             # Update the matched person's center and last seen time
             person_history[matched_id]['center'] = center
             person_history[matched_id]['last_seen'] = current_time
             # print(f"Matched person ID: {matched_id}") # Debug)

        # Get the current state for this person
        person_data = person_history[matched_id]

        # 5. Determine if currently in a "Potential Fall Pose"
        # A pose is potential fall if AR is low OR angle is low (and angle was calculable)
        is_potential_fall_pose_this_frame = False
        if aspect_ratio < HORIZONTAL_AR_THRESHOLD:
             is_potential_fall_pose_this_frame = True
             # print(f"Person {matched_id}: Low AR ({aspect_ratio:.2f})") # Debug)

        if angle is not None and angle < ANGLE_THRESHOLD:
             is_potential_fall_pose_this_frame = True
             # print(f"Person {matched_id}: Low Angle ({angle:.2f})") # Debug)

        # 6. Update State and Timer based on Potential Fall Pose
        if is_potential_fall_pose_this_frame:
            if not person_data['is_potential_fall_pose']:
                # Transition from normal to potential fall pose
                person_data['is_potential_fall_pose'] = True
                person_data['fall_pose_start_time'] = current_time
                # print(f"Person {matched_id}: Started potential fall pose at {current_time:.2f}") # Debug)
        else:
            # Not in a potential fall pose this frame
            if person_data['is_potential_fall_pose']:
                # Transition from potential fall pose to normal
                person_data['is_potential_fall_pose'] = False
                person_data['fall_pose_start_time'] = None
                # print(f"Person {matched_id}: Exited potential fall pose") # Debug)

        # 7. Check for Fall Detection based on Duration
        fall_detected_for_person = False
        if person_data['is_potential_fall_pose'] and person_data['fall_pose_start_time'] is not None:
             duration = current_time - person_data['fall_pose_start_time']
             # print(f"Person {matched_id}: Potential fall duration: {duration:.2f}s") # Debug)
             if duration >= MIN_FALL_POSE_DURATION:
                 fall_detected_for_person = True
                 # print(f"Person {matched_id}: >>> FALL DETECTED! <<<") # Debug)

        person_data['fall_detected'] = fall_detected_for_person # Update the fall status for this person

        # 8. Drawing Results
        color = (0, 0, 255) if person_data['fall_detected'] else (0, 255, 0) # Red for Fall, Green for Normal

        # Add a warning color for the state leading up to a fall
        if person_data['is_potential_fall_pose'] and not person_data['fall_detected']:
             color = (0, 165, 255) # Orange/Yellow for "Potential Fall Pose" state

        label = f"ID {matched_id}: {'Fall' if person_data['fall_detected'] else 'Normal'}"
        # Optionally add duration or angle/AR info for debugging/visualization
        if person_data['is_potential_fall_pose'] and person_data['fall_pose_start_time'] is not None:
             duration_str = f"({current_time - person_data['fall_pose_start_time']:.1f}s)"
             label += " " + duration_str
        # if angle is not None:
        #      label += f" Angle={angle:.0f}"
        # label += f" AR={aspect_ratio:.1f}" # Always show AR

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Ensure text position is not out of bounds
        text_y_pos = max(y1 - 10, 20) # Put text 10px above box, but at least 20px from top
        cv2.putText(frame, label, (x1, text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Optional: Draw keypoints for visualization/debugging
        for name, kpt_idx in ANGLE_KPTS_INDICES.items():
             if kpt_idx < len(keypoints) and keypoints[kpt_idx][0] > 0 and keypoints[kpt_idx][1] > 0:
                  cv2.circle(frame, (int(keypoints[kpt_idx][0]), int(keypoints[kpt_idx][1])), 5, (255, 0, 255), -1) # Purple dots
                  # Add text label for keypoint name
                  cv2.putText(frame, name, (int(keypoints[kpt_idx][0]), int(keypoints[kpt_idx][1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


    # --- History Cleanup ---
    # Remove persons from history who haven't been seen for a while
    if current_time - last_cleanup_time > HISTORY_CLEANUP_INTERVAL:
        ids_to_remove = [
            pid for pid, data in person_history.items()
            if current_time - data['last_seen'] > HISTORY_CLEANUP_INTERVAL
        ]
        for pid in ids_to_remove:
            # print(f"Removing person {pid} from history (not seen for > {HISTORY_CLEANUP_INTERVAL}s)") # Debug
            del person_history[pid]
        last_cleanup_time = current_time


    # --- Display Frame ---
    cv2.imshow("Fall Detection (Combined AR + Angle + Duration)", frame)

    

    # --- Exit Condition ---
    # Press ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("Mean of calculated angles:", np.mean(calculated_angles) if calculated_angles else "No angles calculated")
        break

# --- Release Resources ---
cap.release()
cv2.destroyAllWindows()
print("Mean of calculated angles:", np.mean(calculated_angles) if calculated_angles else "No angles calculated")