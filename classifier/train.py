import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Paths to training and testing directories
train_dir = 'path_to_train_folder'
test_dir = 'path_to_test_folder'

# Load images function (similar to the previous code)
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Load data
X_train_images, y_train_labels = load_images_from_folder(train_dir)
X_test_images, y_test_labels = load_images_from_folder(test_dir)

# Preprocess data
X_train = X_train_images.reshape(len(X_train_images), -1) / 255.0
X_test = X_test_images.reshape(len(X_test_images), -1) / 255.0
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(clf, 'classifier/car_logo_rf_model.joblib')
joblib.dump(le, 'classifier/label_encoder.joblib')
