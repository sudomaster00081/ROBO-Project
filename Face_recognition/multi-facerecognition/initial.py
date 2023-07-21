from typing import List
import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import os
from collections import Counter
import dlib


####LOADINGSCREEN
from tqdm import tqdm
import pyttsx3


#TEXT TO SPEECH
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

    # Load face detector and landmarks predictor from dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor_path = "Face_recognition/MIX/shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(shape_predictor_path)
# person_names = []


def eye_aspect_ratio(landmarks):
    landmarks_array = np.array([[landmark.x, landmark.y] for landmark in landmarks.parts()])

    left_eye = landmarks_array[36:42]  # Extract the landmarks for the left eye
    right_eye = landmarks_array[42:48]  # Extract the landmarks for the right eye

    # Calculate eye aspect ratio for the left eye
    left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
    left_eye_height = np.linalg.norm((left_eye[1] + left_eye[2]) / 2 - left_eye[4])
    left_ear = left_eye_height / left_eye_width

    # Calculate eye aspect ratio for the right eye
    right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
    right_eye_height = np.linalg.norm((right_eye[1] + right_eye[2]) / 2 - right_eye[4])
    right_ear = right_eye_height / right_eye_width

    # Average the eye aspect ratios of both eyes
    ear = (left_ear + right_ear) / 2.0
    return ear





class MpDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

    def detect(self, image, bgr=False):
        if bgr:
            image = image[:, :, ::-1]
        image_rows, image_cols, _ = image.shape
        detections = self.detector.process(image).detections
        if not detections:
            return False, None, None, None
        locations = detections[0].location_data.relative_bounding_box
        start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin, locations.ymin, image_cols, image_rows)
        end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin + locations.width, locations.ymin + locations.height, image_cols, image_rows)
        if (not start_point) or (not end_point):
            return False, None, None, None
        return True, image[start_point[1]:end_point[1], start_point[0]:end_point[0]], start_point[0], start_point[1]


def generate_embedding(cropped_image, bgr=False):
    if bgr:
        cropped_image = cropped_image[:, :, ::-1]
    height, width, _ = cropped_image.shape
    return face_recognition.face_encodings(cropped_image, known_face_locations=[(0, width, height, 0)])[0]


def load_known_faces():
    
    face_embeddings_path = "Face_recognition\multi-facerecognition\face_embeddings.npz"


    known_face_data = np.load(face_embeddings_path)
    known_face_embeddings = known_face_data['embeddings']
    known_face_names = known_face_data['names']
    return known_face_embeddings, known_face_names


def identify_faces(known_face_embeddings, known_face_names, image):
    detector = MpDetector()
    face_detection_status, face_crop, start_x, start_y = detector.detect(image, True)
    if face_detection_status:
        current_face_embedding = generate_embedding(np.array(face_crop))
        face_distances = face_recognition.face_distance(known_face_embeddings, current_face_embedding)
        min_distance_index = np.argmin(face_distances)
        min_distance = face_distances[min_distance_index]
        if min_distance < 0.5:
            # Face recognized as a known person
            print(min_distance)
            return True, known_face_names[min_distance_index], start_x, start_y
        else:
            # Unknown face
            # print ("Unknown")
            unk = "Unknown"
            return True, unk, None, None
    else:
        # No face detected
        print("Noface" , end = "r")
        return False, None, None, None

#COUNT PART
def find_largest_repeating(names):
    counts = Counter(names)
    print(counts)
    max_name, max_count = counts.most_common(1)[0]
    # print(max_name,max_count)
    
    if max_name == 'Unknown' and max_count == 20:
        print ("\nUNKNOWN PERSON Welcome\n")
        # text_to_speech("UNKNOWN PERSON")
        return(None)
    # Check if the highest count is greater than 15
    if max_count > 15 and max_name != 'Unknown':
        accuracyrate = max_count * 5
        print(f"\n\nPerson Identified as : '{max_name}' With Accuracy {accuracyrate} %.\n")
        
        # text_to_speech(f"Person Identified as :{max_name} With Accuracy {accuracyrate} Percentage")
        return (max_name)
        # exit()
    else:
        print("\n\nPerson Unidentified-----Please Come Closer :\n")
        text_to_speech("Person Unidentified-----Please Come Closer :")
        return ("interrupt")



# Function to identify faces and count the number of faces looking at the camera
def identify_faces_with_ear(known_face_embeddings, known_face_names, image):
    detector = MpDetector()
    face_detection_status, face_crop, start_x, start_y = detector.detect(image, True)
    if face_detection_status:
        # Known person recognized
        current_face_embedding = generate_embedding(np.array(face_crop))
        face_distances = face_recognition.face_distance(known_face_embeddings, current_face_embedding)
        min_distance_index = np.argmin(face_distances)
        min_distance = face_distances[min_distance_index]
        if min_distance < 0.5:
            # Face recognized as a known person
            return True, known_face_names[min_distance_index], start_x, start_y
        else:
            # Unknown face
            unk = "Unknown"
            return True, unk, None, None
    else:
        # No face detected
        return False, None, None, None




# Update the main loop with EAR functionality and person count check
def main3():
    known_face_embeddings, known_face_names = load_known_faces()
    cap = cv2.VideoCapture(0)

    prev_x, prev_y = None, None
    person_names = []
    i = 0

    while i < 20:
        ret, frame = cap.read()
        if not ret:
            break

        # Check EAR for each face in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        faces_looking_at_camera = 0

        # Iterate over detected faces
        for face in faces:
            # Predict face landmarks
            landmarks = landmark_predictor(gray, face)

            # Determine if the person is looking at the camera
            ear = eye_aspect_ratio(landmarks)

            # Threshold to consider if the person is looking at the camera
            if ear >= 0.2:  # You can adjust the threshold as per your requirement
                faces_looking_at_camera += 1

        # If more than one face is looking at the camera, print "Hi all" and continue
        if faces_looking_at_camera > 1:
            print("Hi all")
            continue

        # If only one face is looking at the camera, perform face recognition
        if faces_looking_at_camera == 1:
            face_recognition_status, person_name, start_x, start_y = identify_faces_with_ear(
                known_face_embeddings, known_face_names, frame)

            if face_recognition_status:
                person_names.append(person_name)
                i += 1

        prev_x, prev_y = start_x, start_y

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"Recognizing....: {i * 5} %", end="\r")

    person = find_largest_repeating(person_names)
    if person == "interrupt":
        cap.release()
        cv2.destroyAllWindows()
        return main3()

    cap.release()
    cv2.destroyAllWindows()
    return person


if __name__ == "__main__":
    main3()
# print(person_names)
print("\nFingers Are Crossed....🤞")