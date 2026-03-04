import pandas as pd
import joblib
import os
import requests

# ---------------------------------------------------------------
# HuggingFace Model Download
# ---------------------------------------------------------------
HF_BASE_URL = "https://huggingface.co/sridharan-cdm/nutrivision-models/resolve/main"

TARGETS = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'vitamin_c', 'sugar']

def _download_file(url, local_path):
    """Downloads a file from a URL if it doesn't already exist locally."""
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

def _ensure_models_downloaded():
    """Downloads all .joblib models from HuggingFace if not present locally."""
    for target in TARGETS:
        local_path = f'models/{target}_model.joblib'
        if not os.path.exists(local_path):
            try:
                url = f"{HF_BASE_URL}/models/{target}_model.joblib"
                print(f"Downloading {target}_model.joblib from HuggingFace...")
                _download_file(url, local_path)
            except Exception as e:
                print(f"Warning: Could not download {target}_model.joblib: {e}")

# --- Load Trained Models ---
# This dictionary will hold our trained model pipelines.
MODELS = {}

# Download from HuggingFace if not already present
_ensure_models_downloaded()

for target in TARGETS:
    model_path = f'models/{target}_model.joblib'
    if os.path.exists(model_path):
        MODELS[target] = joblib.load(model_path)

def predict_daily_goals(user_profile):
    """
    Predicts daily nutritional goals using pre-trained machine learning models.

    Args:
        user_profile (dict): A dictionary containing 'age', 'gender', 'weight', 'activity_level'.

    Returns:
        dict: A dictionary of predicted daily nutritional goals, or None if models are not loaded.
    """
    if not MODELS:
        print("Warning: Models are not loaded. Run train_goal_predictor.py first.")
        return None # Or return a default dictionary

    # Convert user profile into a DataFrame for prediction
    input_df = pd.DataFrame([user_profile])

    predicted_goals = {}
    for target, model in MODELS.items():
        prediction = model.predict(input_df)[0]
        predicted_goals[target] = round(prediction)

    return predicted_goals