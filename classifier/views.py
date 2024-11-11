import os
from django.shortcuts import render
from django.conf import settings
import joblib
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from django.core.files.storage import default_storage

# Load model and label encoder at the start
MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'MLModel', 'car_logo_rf_model.joblib')
ENCODER_PATH = os.path.join(settings.MEDIA_ROOT, 'MLModel', 'label_encoder.joblib')

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def preprocess_image(image_path):
    """Function to preprocess the image before prediction."""
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to match model's expected input size
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, -1)  # Flatten for Random Forest input
    return img_array

def upload_image(request):
    """View to handle image upload and prediction."""
    if request.method == 'POST' and request.FILES['logo']:
        # Save uploaded file
        logo = request.FILES['logo']
        file_name = logo.name
        upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', file_name)
        # Ensure the file is saved securely
        # path = default_storage.save(upload_path, logo)
            # Ensure the directory exists
        upload_dir = os.path.dirname(upload_path)
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Save the file
        with open(upload_path, 'wb') as f:
            for chunk in logo.chunks():
                f.write(chunk)
        path = upload_path
        # Preprocess the image and make prediction
        image_array = preprocess_image(upload_path)
        prediction = model.predict(image_array)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Render the result page with the prediction
        return render(request, 'classifier/result.html', {'label': predicted_label})

    return render(request, 'classifier/upload.html')
