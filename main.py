import face_recognition
import cv2
import os
import csv
from datetime import datetime
import uuid  # for unique unknown filenames

# Load known faces
known_faces = []
known_face_encodings = []
known_ages = []

# Directories
known_faces_dir = "images/known_faces/"
unknown_faces_dir = "images/unknown_faces/"

# Create unknown_faces directory if it doesn't exist
os.makedirs(unknown_faces_dir, exist_ok=True)

# Load the known faces and their names and ages
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name_age = filename.split('.')[0]  # remove file extension
        if '_' in name_age:
            name, age = name_age.split('_')
        else:
            name, age = name_age, "Unknown"
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(name)
        known_ages.append(age)
        known_face_encodings.append(encoding)

# Load webcam
video_capture = cv2.VideoCapture(0)

# Attendance file
attendance_file = 'attendance.csv'

# Check if attendance file exists, if not, create it
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# Track people who have already been marked as "Present" during the session
attended_names = set()

# Frame processing control (process every 5th frame)
frame_counter = 0

# Main loop
while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Skip processing every frame, only process every 5th frame for performance
    frame_counter += 1
    if frame_counter % 5 != 0:
        continue

    # Resize the frame to speed up face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Loop through each face found
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        age = "Unknown"
        status = "Unknown"  # Variable for showing the status

        if True in matches:
            first_match_index = matches.index(True)
            name = known_faces[first_match_index]
            age = known_ages[first_match_index]
            status = "Present"  # Set status as Present if the face is known

            # Only record if the person hasn't been marked as present yet
            if name not in attended_names:
                attended_names.add(name)  # Mark this person as attended

                # Record attendance
                with open(attendance_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        else:
            # Capture and save unknown face
            top_u, right_u, bottom_u, left_u = [coord * 4 for coord in (top, right, bottom, left)]  # scale back
            unknown_face = frame[top_u:bottom_u, left_u:right_u]

            if unknown_face.size != 0:  # make sure face is detected
                filename = f"{unknown_faces_dir}/unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
                cv2.imwrite(filename, unknown_face)

        # Scale back to original frame size for drawing
        top, right, bottom, left = [int(coord * 4) for coord in (top, right, bottom, left)]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Prepare the display text with status
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        display_text = f"{name}, Age: {age}, {status}, {current_time}"

        # Put text above rectangle
        cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
