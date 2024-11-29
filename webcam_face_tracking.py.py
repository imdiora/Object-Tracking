import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Start the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to manage tracking state
tracking = False
bbox = None  # Bounding box for the tracker

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # If we are not currently tracking, detect the face
    if not tracking:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If a face is detected, initialize the tracker
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the first detected face
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)
            tracking = True
        else:
            # Display "Lost Tracking" if no face is detected
            cv2.putText(frame, "Lost Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Update the tracker
        success, bbox = tracker.update(frame)

        if success:
            # If tracking is successful, draw the bounding box in green
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, "Tracking Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If tracking fails, reset the tracker and display "Lost Tracking"
            tracking = False
            cv2.putText(frame, "Lost Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()