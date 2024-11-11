import joblib
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Load the saved model and label encoder
clf = joblib.load('classifier/car_logo_rf_model.joblib')
le = joblib.load('classifier/label_encoder.joblib')

# Define a function to preprocess and test new images
def test_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Error loading image"
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, -1) / 255.0
    pred = clf.predict(img)
    pred_label = le.inverse_transform(pred)
    return pred_label[0]

# Example usage
print("Predicted label:", test_image("path_to_sample_test_image"))
