import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model   # <- changed
import os

@st.cache_resource
def load_detection_model():
    MODEL_PATH = "model.h5"
    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found locally. If you host model externally, download it here.")
        # optionally download from a URL if you host the model externally
    model = load_model(MODEL_PATH, compile=False)
    return model


SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

def find_medicine(pred):
    if pred == 0:
        return "fluorouracil"
    elif pred == 1:
        return "Aldara"
    elif pred == 2:
        return "Prescription Hydrogen Peroxide"
    elif pred == 3:
        return "fluorouracil"
    elif pred == 4:
        return "fluorouracil (5-FU)"
    elif pred == 5:
        return "fluorouracil"
    elif pred == 6:
        return "fluorouracil"

uploaded_file = st.file_uploader("Upload a skin image for detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect"):
        # Preprocess the image
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        accuracy_percent = round(accuracy * 100, 2)
        medicine = find_medicine(pred)

        st.markdown(f"**Detection Result:** {disease}")
        st.markdown(f"**Confidence:** {accuracy_percent}%")
        st.markdown(f"**Recommended Medicine:** {medicine}")
