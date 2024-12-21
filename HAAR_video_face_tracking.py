import cv2
import numpy as np
import time

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Path to video file
video_path = "di.mp4"
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

def detect_people(frame):
    start_time = time.time()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces and bodies
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    bodies = body_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    
    detections = []
    
    # Process detections
    for (x, y, w, h) in faces:
        detections.append((x, y, w, h))
    for (x, y, w, h) in bodies:
        detections.append((x, y, w, h))
    
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return detections, processing_time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    
    # Get detections and processing time
    detections, proc_time = detect_people(frame)
    processing_times.append(proc_time)
    
    # Calculate metrics
    detected_count = len(detections)
    current_accuracy = (min(detected_count, expected_people) / expected_people) * 100
    accuracy_rates.append(current_accuracy)
    
    # Calculate averages
    avg_proc_time = sum(processing_times) / len(processing_times)
    avg_accuracy = sum(accuracy_rates) / len(accuracy_rates)
    
    if detected_count > 0:
        successful_detections += 1
    frame_success_rate = (successful_detections / total_frames) * 100
    
    # Draw rectangles and labels
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Person Detected', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display metrics
    metrics_text = [
        f'People Detected: {detected_count}',
        f'Current Processing Time: {proc_time:.2f}ms',
        f'Average Processing Time: {avg_proc_time:.2f}ms',
        f'Current Accuracy: {current_accuracy:.1f}%',
        f'Average Accuracy: {avg_accuracy:.1f}%',
        f'Frame Success Rate: {frame_success_rate:.1f}%'
    ]
    
    for i, text in enumerate(metrics_text):
        cv2.putText(frame, text, (10, 30 + (30 * i)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('People Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final statistics
print("\nFinal Statistics:")
print(f"Total Frames Processed: {total_frames}")
print(f"Average Processing Time: {avg_proc_time:.2f}ms")
print(f"Average Detection Accuracy: {avg_accuracy:.1f}%")
print(f"Overall Frame Success Rate: {frame_success_rate:.1f}%")

cap.release()
cv2.destroyAllWindows()
