import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import pandas as pd

# --- Data Loading Functions ---

@st.cache_data
def load_nutrition_data():
    """Loads nutritional data from a CSV file."""
    df = pd.read_csv('nutrition.csv')
    df.set_index('food', inplace=True)
    return df

@st.cache_resource
def load_model_and_class_names():
    """Loads the trained model and class names."""
    try:
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
