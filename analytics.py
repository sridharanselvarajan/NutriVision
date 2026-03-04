import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def get_food_log_analytics(food_log, nutrition_df):
    """
    Calculates basic analytics from the user's food log.

    Returns:
        - A Plotly figure for the most frequent foods.
        - A Plotly figure for the cumulative calorie trend.
    """
    if not food_log:
        return None, None

    log_df = pd.DataFrame(food_log, columns=['food'])

    # Most frequent foods
    top_foods = log_df['food'].value_counts().head(5)
    fig_top_foods = px.bar(top_foods, x=top_foods.index, y=top_foods.values,
                           labels={'x': 'Food', 'y': 'Count'}, title="Most Eaten Foods")

    # Cumulative calorie trend
    calorie_log = [nutrition_df.loc[food, 'calories'] for food in food_log]
    cumulative_calories = pd.Series(calorie_log).cumsum()
    fig_cal_trend = px.line(x=range(1, len(cumulative_calories) + 1), y=cumulative_calories,
                            labels={'x': 'Food Item Added', 'y': 'Cumulative Calories'}, title="Calorie Intake Trend")

    return fig_top_foods, fig_cal_trend

def get_diet_diversity_insights(food_log, nutrition_df, n_clusters=3):
    """
    Uses KMeans clustering to analyze diet diversity and provide insights.
    """
    if len(set(food_log)) < n_clusters:
        return "Log more items to get diet diversity insights.", None

    # 1. Prepare data for clustering
    eaten_foods_df = nutrition_df.loc[list(set(food_log))].copy()
    nutrient_cols = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber']
    X = eaten_foods_df[nutrient_cols]

    # 2. Scale data and apply KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    eaten_foods_df['cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Analyze the dominant cluster in the user's log
    log_df = pd.DataFrame(food_log, columns=['food'])
    log_with_clusters = log_df.merge(eaten_foods_df[['cluster']], left_on='food', right_index=True)
    dominant_cluster = log_with_clusters['cluster'].mode()[0]

    # 4. Generate Insight
    cluster_foods = eaten_foods_df[eaten_foods_df['cluster'] == dominant_cluster].index.tolist()
    insight = f"💡 **AI Insight:** Your diet is currently focused on a group of foods including **{', '.join(cluster_foods)}**. Consider diversifying by adding items with different nutritional profiles."

    return insight, eaten_foods_df[['cluster'] + nutrient_cols]