import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def recommend_substitutes(food_name, nutrition_df, num_neighbors=10, num_results=3):
    """
    Recommends healthier substitutes for a given food using KNN for similarity
    and then filtering based on health criteria (lower calories and sugar).

    Args:
        food_name (str): The name of the food to find a substitute for.
        nutrition_df (pd.DataFrame): The dataframe with nutritional info.
        num_neighbors (int): The number of similar items to initially find.
        num_results (int): The final number of substitutes to return.

    Returns:
        pd.DataFrame: A dataframe of healthier substitutes, or None.
    """
    if food_name not in nutrition_df.index:
        return None

    # 1. Feature Engineering & Normalization
    # Use all numeric columns for finding nutritional similarity
    nutrient_cols = nutrition_df.select_dtypes(include='number').columns
    X = nutrition_df[nutrient_cols]

    # Normalize the data so all nutrients have a similar scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. ML Model for Similarity (KNN)
    # Use cosine distance as it's good for finding items with similar proportions
    knn = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    knn.fit(X_scaled)

    # 3. Find Similar Items
    # Get the index and scaled vector for the input food
    food_index = nutrition_df.index.get_loc(food_name)
    food_vector = X_scaled[food_index].reshape(1, -1)

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(food_vector)

    # Get the names of the most similar foods (excluding the food itself)
    similar_foods_indices = indices.flatten()[1:]
    similar_foods = nutrition_df.iloc[similar_foods_indices]

    # 4. Filtering Layer for "Healthier" Options
    original_food_stats = nutrition_df.loc[food_name]
    healthier_substitutes = similar_foods[
        (similar_foods['calories'] < original_food_stats['calories']) &
        (similar_foods['sugar'] < original_food_stats['sugar'])
    ]

    # Return the top N results
    return healthier_substitutes.head(num_results)