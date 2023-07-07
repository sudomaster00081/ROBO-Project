import cv2
import numpy as np
import face_recognition
import joblib

# Load the trained face recognition model
svm_classifier = joblib.load("face_recognition_model.joblib")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load face detection model from the face_recognition library
face_detection_model = "cnn"  # or "hog"
face_detection = face_recognition.api.face_detection.FaceDetectorModel(face_detection_model)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Detect faces in the frame
    face_locations = face_detection.detect_faces(frame)

    # Iterate over detected faces
    for face_location in face_locations:
        # Extract face coordinates
        top, right, bottom, left = face_location["box"]

        # Crop the face region from the frame
        face_image = frame[top:bottom, left:right]

        # Resize the face image to a fixed size for face recognition
        face_image = cv2.resize(face_image, (128, 128))

        # Convert BGR image to RGB (face_recognition library requires RGB format)
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Generate face embedding for the current face
        face_embedding = face_recognition.face_encodings(face_image_rgb)

        # If at least one face embedding is found
        if len(face_embedding) > 0:
            # Recognize the face using the trained SVM classifier
            predicted_person = svm_classifier.predict(face_embedding)
            confidence = svm_classifier.decision_function(face_embedding)

            # Draw bounding box and display the recognized person's name on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_person[0]} ({confidence[0]:.2f})", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with recognized faces
    cv2.imshow("Face Recognition", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
