import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import os

def generate_synthetic_data(num_samples=5000):
    """Generates a synthetic dataset of user profiles and nutritional goals."""
    # Generate random user profiles
    ages = np.random.randint(18, 80, num_samples)
    genders = np.random.choice(['Male', 'Female'], num_samples)
    weights = np.random.uniform(50, 120, num_samples)
    activity_levels = np.random.choice(['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'], num_samples)

    data = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'weight': weights,
        'activity_level': activity_levels
    })

    # Calculate goals using formulas (to create our "ground truth")
    def calculate_goals(row):
        if row['gender'] == 'Male':
            base_calories = (10 * row['weight']) + (6.25 * 170) - (5 * row['age']) + 5
            vitamin_c = 90
        else:
            base_calories = (10 * row['weight']) + (6.25 * 160) - (5 * row['age']) - 161
            vitamin_c = 75

        activity_multipliers = {'Sedentary': 1.2, 'Lightly Active': 1.375, 'Moderately Active': 1.55, 'Very Active': 1.725}
        calories = base_calories * activity_multipliers[row['activity_level']]
        protein = row['weight'] * 1.6
        carbs = (calories * 0.45) / 4
        fat = (calories * 0.25) / 9
        sugar = (calories * 0.1) / 4 # Recommended max 10% of calories from sugar
        fiber = 14 * (calories / 1000)
        
        return pd.Series([calories, protein, carbs, fat, fiber, vitamin_c, sugar])

    data[['calories', 'protein', 'carbs', 'fat', 'fiber', 'vitamin_c', 'sugar']] = data.apply(calculate_goals, axis=1)
    return data

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()

    features = ['age', 'gender', 'weight', 'activity_level']
    targets = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'vitamin_c', 'sugar']

    X = df[features]

    # Define the preprocessing pipeline for our features
    # This will one-hot encode categorical features and leave numerical features as is.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'activity_level'])
        ],
        remainder='passthrough'
    )

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Train a separate model for each target nutrient
    for target in targets:
        print(f"Training model for: {target}...")
        y = df[target]
        
        # Create a full pipeline including preprocessing and the model
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
        model_pipeline.fit(X, y)
        
        # Save the entire pipeline
        joblib.dump(model_pipeline, f'models/{target}_model.joblib')
        print(f"Saved model to models/{target}_model.joblib")