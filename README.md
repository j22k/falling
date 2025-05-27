# Fall Detection System

This project implements a real-time fall detection system using computer vision and pose estimation. The system utilizes YOLOv8 for human pose detection and implements multiple criteria for fall detection.

## Features

- Real-time human pose detection using YOLOv8
- Multi-criteria fall detection:
  - Body pose aspect ratio analysis
  - Hip angle measurement
  - Duration-based confirmation
  - Full body visibility filtering
- Person tracking across video frames
- Visual status indicators (Green: Normal, Orange: Potential Fall, Red: Fall Detected)
- Support for various video sources (video files, webcam, RTSP streams)

## Dependencies

- Python 3.x
- OpenCV (cv2)
- Ultralytics YOLO
- NumPy

## Installation

1. Install the required packages:
```bash
pip install ultralytics opencv-python numpy
```

2. Download the YOLOv8 pose estimation model:
- The code uses `yolov8n-pose.pt` by default
- Alternatively, you can use `yolov8m-pose.pt` or `yolov8l-pose.pt` for better accuracy

## Configuration

The system includes several configurable parameters:

- Detection thresholds
- Minimum confidence levels
- Fall detection criteria
- Tracking parameters

These can be adjusted in the constants section at the top of `final.py`.

## Usage

1. Place your video file in the project directory or configure the video source
2. Run the script:
```bash
python final.py
```
3. Press ESC to exit the program

## Demo

https://github.com/j22k/falling/tree/main/assets/demo.mp4

<video width="100%" controls>
  <source src="fall_detection_output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Watch the demo video above to see the fall detection system in action. The video demonstrates:
- Real-time pose detection and tracking
- Visual indicators for different states (Normal, Potential Fall, Fall Detected)
- Multiple fall detection scenarios
- Person tracking with unique IDs

## Detection States

- 🟢 Green: Normal pose detected
- 🟠 Orange: Potential fall pose (under observation)
- 🔴 Red: Fall detected and confirmed

## How It Works

1. **Pose Detection**: Uses YOLOv8 to detect human poses in each frame
2. **Fall Criteria**:
   - Aspect Ratio: Detects horizontal body positions
   - Hip Angle: Measures the angle between shoulder, hip, and knee
   - Duration: Confirms falls based on sustained unusual poses
3. **Person Tracking**: Maintains identity of detected persons across frames
4. **Full Body Filtering**: Ensures reliable detection by tracking full body visibility

## License

This project is intended for research and development purposes. Use responsibly.

## Output Example

The system generates an annotated video output (`fall_detection_output.mp4`) showing:
- Pose keypoints
- Bounding boxes
- Person IDs
- Fall status indicators
- Duration timers for potential falls
