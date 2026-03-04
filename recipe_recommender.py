import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recipe_data import RECIPE_DATABASE

def _vectorize_recipes(nutrition_df):
    """
    Converts each recipe in the database into a nutritional vector.

    Args:
        nutrition_df (pd.DataFrame): The dataframe with nutritional info for single ingredients.

    Returns:
        pd.DataFrame: A dataframe where each row is a recipe and columns are its total nutrients.
    """
    nutrient_cols = nutrition_df.columns
    recipe_vectors = []

    for recipe in RECIPE_DATABASE:
        # Sum the nutrient vectors of all ingredients in the recipe
        recipe_vector = nutrition_df.loc[recipe["ingredients"]].sum()
        recipe_info = {
            "name": recipe["name"],
            "ingredients_list": ", ".join([i.title() for i in recipe["ingredients"]]),
            "url": recipe["url"]
        }
        # Combine recipe info with its nutrient vector
        recipe_info.update(recipe_vector.to_dict())
        recipe_vectors.append(recipe_info)

    return pd.DataFrame(recipe_vectors).set_index('name')

def recommend_recipes(recognized_food, current_intake, daily_goals, nutrition_df, num_results=3):
    """
    Recommends recipes containing a specific food, ranked by nutritional similarity to user's needs.
    """
    # 1. Vectorize all recipes based on their ingredients' nutrients
    recipe_vectors_df = _vectorize_recipes(nutrition_df)

    # 2. Filter for recipes that contain the recognized food
    # We search the stringified 'ingredients_list' column
    relevant_recipes = recipe_vectors_df[
        recipe_vectors_df['ingredients_list'].str.contains(recognized_food.title(), case=False)
    ]

    if relevant_recipes.empty:
        return None

    # 3. Calculate the "ideal" remaining nutrients needed (same as in single food recommender)
    nutrient_cols = nutrition_df.columns
    goal_vector = np.array([daily_goals.get(col, 0) for col in nutrient_cols])
    current_vector = np.array([current_intake.get(col, 0) for col in nutrient_cols])
    ideal_remaining_vector = (goal_vector - current_vector).clip(0, None).reshape(1, -1)

    # 4. Calculate Cosine Similarity between user's needs and each recipe's nutritional profile
    recipe_nutrient_vectors = relevant_recipes[nutrient_cols].values
    similarities = cosine_similarity(ideal_remaining_vector, recipe_nutrient_vectors)

    # 5. Get top N recommendations
    top_indices = similarities[0].argsort()[-num_results:][::-1]
    recommended_recipes = relevant_recipes.iloc[top_indices]

    return recommended_recipes[['ingredients_list', 'url']]