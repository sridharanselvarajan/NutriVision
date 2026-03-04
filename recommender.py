import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommender(current_intake, daily_goals, nutrition_df, food_log):
    """
    Recommends foods using a content-based approach with cosine similarity.
    """
    # Define the nutrient columns based on the main nutrition dataframe
    nutrient_cols = nutrition_df.columns.tolist()

    # 1. Calculate the "ideal" remaining nutrients needed
    goal_vector = np.array([daily_goals.get(col, 0) for col in nutrient_cols])
    current_vector = np.array([current_intake.get(col, 0) for col in nutrient_cols])
    # Use clip(0, None) to ensure we don't have negative needs
    ideal_remaining_vector = (goal_vector - current_vector).clip(0, None)

    # If goals are mostly met, don't recommend anything
    if np.sum(ideal_remaining_vector) < 10:  # Threshold to stop recommending
        return None

    # 2. Prepare food vectors
    # Exclude foods already in the log
    available_foods_df = nutrition_df.loc[~nutrition_df.index.isin(food_log)]
    if available_foods_df.empty:
        return None
        
    # Ensure the columns match the order of our goal/intake vectors
    food_vectors = available_foods_df[nutrient_cols].values

    # 3. Calculate Cosine Similarity
    ideal_remaining_vector = ideal_remaining_vector.reshape(1, -1)
    similarities = cosine_similarity(ideal_remaining_vector, food_vectors)

    # 4. Get top 3 recommendations
    top_indices = similarities[0].argsort()[-3:][::-1]
    recommended_foods = available_foods_df.iloc[top_indices]

    return recommended_foods
