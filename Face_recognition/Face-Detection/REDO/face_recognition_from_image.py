import cv2
import numpy as np
import face_recognition
import joblib

# Load the trained face recognition model
svm_classifier = joblib.load("face_recognition_model.joblib")

# Load the image for face recognition
image_path = "image.jpg"
image = cv2.imread(image_path)

# Convert BGR image to RGB (face_recognition library requires RGB format)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces in the image
face_locations = face_recognition.face_locations(rgb_image)

# Iterate over detected faces
for top, right, bottom, left in face_locations:
    # Crop the face region from the image
    face_image = image[top:bottom, left:right]

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

        # Print the identified person's name
        print("Identified person:", predicted_person[0])

