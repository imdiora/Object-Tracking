import cv2
import numpy as np
import time

# Path to video file
video_path = "small_instance.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Variables for metrics
total_frames = 0
successful_detections = 0
expected_people = 3  # Number of golfers in the scene
processing_times = []
accuracy_rates = []

# Initialize trackers list
trackers = []
detection_interval = 30  # Detect faces every 30 frames

def create_tracker():
    return cv2.legacy.TrackerCSRT_create()

def detect_people(frame):
    start_time = time.time()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    
    processing_time = (time.time() - start_time) * 1000
    return faces, processing_time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    start_time = time.time()
    
    # Detect faces every N frames or when no trackers exist
    if total_frames % detection_interval == 0 or len(trackers) == 0:
        faces, detection_time = detect_people(frame)
        
        # Clear existing trackers
        trackers = []
        
        # Initialize new trackers for each detected face
        for (x, y, w, h) in faces:
            tracker = create_tracker()
            tracker.init(frame, (x, y, w, h))
            trackers.append({'tracker': tracker, 'bbox': (x, y, w, h)})
    
    # Update existing trackers
    detected_count = 0
    for tracker_info in trackers[:]:
        success, bbox = tracker_info['tracker'].update(frame)
        if success:
            detected_count += 1
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person Tracked', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            tracker_info['bbox'] = bbox
    
    # Calculate metrics
    proc_time = (time.time() - start_time) * 1000
    processing_times.append(proc_time)
    
    current_accuracy = (min(detected_count, expected_people) / expected_people) * 100
    accuracy_rates.append(current_accuracy)
    
    # Calculate averages
    avg_proc_time = sum(processing_times) / len(processing_times)
    avg_accuracy = sum(accuracy_rates) / len(accuracy_rates)
    
    if detected_count > 0:
        successful_detections += 1
    frame_success_rate = (successful_detections / total_frames) * 100
    
    # Display metrics
    metrics_text = [
        f'People Tracked: {detected_count}',
        f'Current Processing Time: {proc_time:.2f}ms',
        f'Average Processing Time: {avg_proc_time:.2f}ms',
        f'Current Accuracy: {current_accuracy:.1f}%',
        f'Average Accuracy: {avg_accuracy:.1f}%',
        f'Frame Success Rate: {frame_success_rate:.1f}%'
    ]
    
    for i, text in enumerate(metrics_text):
        cv2.putText(frame, text, (10, 30 + (30 * i)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('People Tracking (CSRT)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nFinal Statistics:")
print(f"Total Frames Processed: {total_frames}")
print(f"Average Processing Time: {avg_proc_time:.2f}ms")
print(f"Average Detection Accuracy: {avg_accuracy:.1f}%")
print(f"Overall Frame Success Rate: {frame_success_rate:.1f}%")

cap.release()
cv2.destroyAllWindows()
