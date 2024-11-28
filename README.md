# Object-Tracking
# Face Detection and Tracking with OpenCV

This repository contains two Python programs for face detection and tracking using OpenCV. The first program uses a webcam for real-time face tracking, while the second processes a video file to detect and track faces. Both programs utilize Haar Cascade for face detection and the CSRT tracking algorithm for robust face tracking.

---

## Features

- **Face Detection**: Detects faces using OpenCV's Haar Cascade classifier.
- **Face Tracking**: Tracks the detected face using the CSRT (Channel and Spatial Reliability Tracker) algorithm.
- **Dynamic Status Updates**:
  - Displays "Tracking Face" when the face is successfully tracked.
  - Displays "Lost Tracking" when the face is not detected or tracking fails.
- **Input Options**:
  - Webcam input for real-time face tracking.
  - Video file input for processing pre-recorded videos.

---

## Requirements

To run the programs, you need the following:

- Python 3.x
- OpenCV library (with contrib modules for tracking algorithms)

Install OpenCV using pip:
```bash
pip install opencv-python opencv-contrib-python

