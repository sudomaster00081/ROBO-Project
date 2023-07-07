import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the labeled dataset (face embeddings and person names)
data = np.load("face_embeddings.npz")
face_embeddings = data["embeddings"]
person_names = data["names"]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(face_embeddings, person_names, test_size=0.2, random_state=42)

# Create an SVM classifier and train it
svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = svm_classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(svm_classifier, "face_recognition_model.joblib")
print("Model saved successfully!")
