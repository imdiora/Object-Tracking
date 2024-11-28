# Object-Tracking
# Face Detection and Tracking with OpenCV

This repository contains two Python programs for real-time face detection and tracking using OpenCV. The first program uses a webcam as the input source, while the second program processes a video file. Both programs utilize OpenCV's Haar Cascade for face detection and the CSRT tracking algorithm for robust face tracking.

---

## Features

- **Face Detection**: Detects faces in real-time using OpenCV's Haar Cascade classifier.
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
How to Use
1. Real-Time Face Tracking with Webcam
This program uses your webcam to detect and track faces in real-time.
Steps:
Run the webcam_face_tracking.py script.
The program will open a window showing the webcam feed.
A green rectangle will appear around the detected face, and the status "Tracking Face" will be displayed.
If the face is lost, the status will change to "Lost Tracking."
Press q to exit the program.
Command:
bash


python webcam_face_tracking.py
2. Face Tracking with Video Input
This program processes a video file to detect and track faces.
Steps:
Replace the video_path variable in the video_face_tracking.py script with the path to your video file.
python


video_path = "input_video.mp4"  # Replace with your video file path
Run the script.
The program will display the video with a green rectangle around the detected face and the status "Tracking Face."
If the face is lost, the status will change to "Lost Tracking."
Press q to exit the program.
Command:
bash


python video_face_tracking.py
File Descriptions
webcam_face_tracking.py:
Tracks faces in real-time using a webcam.
Displays a green bounding box around the detected face and updates the tracking status.
video_face_tracking.py:
Tracks faces in a pre-recorded video file.
Similar functionality to the webcam version but processes video input instead.
Customization
Change Input Source:
For the video version, replace the video_path variable with the path to your video file.
Adjust Detection Parameters:
Modify the scaleFactor and minNeighbors parameters in the detectMultiScale function to fine-tune face detection sensitivity.
python


faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
Change Tracker:
Replace cv2.TrackerCSRT_create() with other OpenCV trackers like cv2.TrackerKCF_create() or cv2.TrackerMIL_create() for different tracking algorithms.
Example Output
Webcam Input:
Tracking Face:
A green rectangle is drawn around the detected face.
Status: "Tracking Face" (in green).
Lost Tracking:
No rectangle is displayed.
Status: "Lost Tracking" (in red).
Video Input:
Similar output as the webcam version but processes frames from a video file.
Notes
Ensure your webcam is functional for the real-time tracking program.
For video input, use a valid video file format (e.g., .mp4, .avi).
The CSRT tracker is robust but may fail in cases of extreme occlusion or rapid movement.
License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code.
Acknowledgments
OpenCV for providing powerful tools for computer vision.
Haar Cascade for face detection.
CSRT tracker for robust face tracking.
Enjoy experimenting with face detection and tracking! 😊
