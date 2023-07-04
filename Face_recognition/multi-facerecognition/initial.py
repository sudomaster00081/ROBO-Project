from typing import List
import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import pyttsx3


# TEXT TO SPEECH
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()


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
    face_embeddings_path = "face_embeddings.npz"
    known_face_data = np.load(face_embeddings_path)
    known_face_embeddings = known_face_data['embeddings']
    known_face_names = known_face_data['names']
    return known_face_embeddings, known_face_names


def identify_faces(known_face_embeddings, known_face_names, image):
    detector = MpDetector()
    face_detection_status, face_crops, start_points_x, start_points_y = detector.detect(image, True)
    identified_faces = []

    if face_detection_status:
        for face_crop, start_x, start_y in zip(face_crops, start_points_x, start_points_y):
            current_face_embedding = generate_embedding(np.array(face_crop))
            face_distances = face_recognition.face_distance(known_face_embeddings, current_face_embedding)
            min_distance_index = np.argmin(face_distances)
            min_distance = face_distances[min_distance_index]

            if min_distance < 0.5:
                # Face recognized as a known person
                identified_faces.append((known_face_names[min_distance_index], start_x, start_y))
            else:
                # Unknown face
                identified_faces.append(('Unknown', None, None))

    return identified_faces


def find_largest_repeating(names):
    counts = Counter(names)
    max_name, max_count = counts.most_common(1)[0]

    if max_name == 'Unknown' and max_count == 20:
        print("\nUNKNOWN PERSON Welcome\n")
        return None

    # Check if the highest count is greater than 15
    if max_count > 15 and max_name != 'Unknown':
        accuracy_rate = max_count * 5
        print(f"\n\nPerson Identified as: '{max_name}' with Accuracy {accuracy_rate}%.\n")
        return max_name
    else:
        print("\n\nPerson Unidentified ----- Please Come Closer:\n")
        return "interrupt"


def main1():
    known_face_embeddings, known_face_names = load_known_faces()
    cap = cv2.VideoCapture(0)

    person_names = []
    i = 0
    while i < 20:
        ret, frame = cap.read()
        if not ret:
            break

        face_recognition_results = identify_faces(known_face_embeddings, known_face_names, frame)

        for person_name, start_x, start_y in face_recognition_results:
            person_names.append(person_name)
            i = i + 1

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"Recognizing....: {i*5} %", end="\r")

    person = find_largest_repeating(person_names)
    if person == "interrupt":
        cap.release()
        cv2.destroyAllWindows()
        return main1()

    cap.release()
    cv2.destroyAllWindows()
    return person


if __name__ == "__main__":
    main1()

print("\nFingers Are Crossed....🤞")
