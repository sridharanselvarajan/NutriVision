import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

# --- Badge Definitions ---
# We define badges with a check function (lambda) that determines if the badge is earned.
BADGE_DEFINITIONS = {
    "Balanced Diet": {
        "icon": "🥦",
        "description": "Ate 5 or more different types of food today.",
        "check": lambda log_df, *args: log_df['food'].nunique() >= 5
    },
    "Protein Power": {
        "icon": "💪",
        "description": "Met your daily protein goal.",
        "check": lambda log_df, intake, goals, *args: intake.get('protein', 0) >= goals.get('protein', 999)
    },
    "Vitamin C Hero": {
        "icon": "🍋",
        "description": "Ate at least 2 high Vitamin C foods ( > 50mg per 100g).",
        "check": lambda log_df, intake, goals, nutrition_df: \
            log_df.merge(nutrition_df[nutrition_df['vitamin_c'] > 50], left_on='food', right_index=True).shape[0] >= 2
    },
    "Hydration Hero": {
        "icon": "💧",
        "description": "Ate a super hydrating food like Watermelon or Cucumber.",
        "check": lambda log_df, *args: any(food in log_df['food'].values for food in ['watermelon', 'cucumber'])
    },
    "Fiber Fanatic": {
        "icon": "🌾",
        "description": "Met your daily fiber goal.",
        "check": lambda log_df, intake, goals, *args: intake.get('fiber', 0) >= goals.get('fiber', 999)
    }
}

def check_badges(food_log, total_intake, daily_goals, nutrition_df):
    """Checks the food log against badge definitions to see what has been earned."""
    if not food_log:
        return []

    earned_badges = []
    log_df = pd.DataFrame(food_log, columns=['food'])

    for name, details in BADGE_DEFINITIONS.items():
        try:
            if details["check"](log_df, total_intake, daily_goals, nutrition_df):
                earned_badges.append({"name": name, "icon": details["icon"], "description": details["description"]})
        except Exception:
            # Fails gracefully if data is missing for a check
            continue
            
    return earned_badges

def generate_personalized_challenge(food_log, nutrition_df, n_clusters=5):
    """
    Uses KMeans clustering to find an under-represented food group in the user's log
    and generates a personalized challenge.
    """
    if not food_log:
        return "Log your first food to receive a personalized challenge!", None, 0

    # 1. ML Model: Cluster ALL foods in the database into nutritional groups
    nutrient_cols = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber']
    X = nutrition_df[nutrient_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    nutrition_df['cluster'] = kmeans.fit_predict(X_scaled)

    # 2. Identify clusters the user has eaten from
    eaten_clusters = set(nutrition_df.loc[list(set(food_log))]['cluster'])

    # 3. Find an under-represented cluster (one the user hasn't eaten from)
    all_clusters = set(range(n_clusters))
    under_represented_clusters = list(all_clusters - eaten_clusters)

    if not under_represented_clusters:
        return "You've had a very diverse diet today! Great job!", None, 100

    # 4. Generate Challenge: Pick a food from an under-represented cluster
    target_cluster = random.choice(under_represented_clusters)
    challenge_options = nutrition_df[nutrition_df['cluster'] == target_cluster].index.tolist()
    challenge_food = random.choice(challenge_options)

    challenge_text = f"Your diet is looking great! To make it even more diverse, try a food from a new nutritional group. **Your challenge: eat a {challenge_food.title()}!**"
    
    # Calculate progress
    progress = (1 - len(under_represented_clusters) / n_clusters) * 100

    return challenge_text, challenge_food, progress