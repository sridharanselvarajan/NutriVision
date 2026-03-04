import pandas as pd
import joblib
import os

# --- Load Trained Models ---
# This dictionary will hold our trained model pipelines.
MODELS = {}
TARGETS = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'vitamin_c', 'sugar']

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