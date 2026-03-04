import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import pandas as pd
import os
import requests

# ---------------------------------------------------------------
# HuggingFace Model Download
# ---------------------------------------------------------------
HF_BASE_URL = "https://huggingface.co/sridharan-cdm/nutrivision-models/resolve/main"

def download_file(url, local_path):
    """Downloads a file from a URL if it doesn't already exist locally."""
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True) if os.path.dirname(local_path) else None
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# --- Data Loading Functions ---

@st.cache_data
def load_nutrition_data():
    """Loads nutritional data from a CSV file."""
    df = pd.read_csv('nutrition.csv')
    df.set_index('food', inplace=True)
    return df

@st.cache_resource
def load_model_and_class_names():
    """Loads the trained model and class names, downloading from HuggingFace if needed."""
    try:
        # Download keras model from HuggingFace if not present locally
        if not os.path.exists('food_classifier.keras'):
            with st.spinner('⬇️ Downloading food classifier model... (first run only, ~25MB)'):
                download_file(f"{HF_BASE_URL}/food_classifier.keras", "food_classifier.keras")

        model = tf.keras.models.load_model('food_classifier.keras')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model or class names: {e}")
        return None, None

# --- Prediction Function ---

def predict(image: Image.Image, model, class_names):
    """Preprocesses an image and returns top 3 predictions."""
    # Resize and convert to array
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)

    # Add batch dimension and preprocess for MobileNetV2
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_array, axis=0))

    # Make prediction
    predictions = model.predict(img_preprocessed)

    # Get top 3 predictions
    top_indices = predictions[0].argsort()[-3:][::-1]
    top_predictions = [(class_names[i], predictions[0][i] * 100) for i in top_indices]
    return top_predictions
