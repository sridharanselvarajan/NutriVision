import streamlit as st
from PIL import Image
import pandas as pd
from utils import load_model_and_class_names, load_nutrition_data, predict
from recommender import content_based_recommender
from meal_planner import suggest_meal_combinations
from recipe_recommender import recommend_recipes
from goal_predictor import predict_daily_goals
from substitute_recommender import recommend_substitutes
from analytics import get_food_log_analytics, get_diet_diversity_insights
from gamification import check_badges, generate_personalized_challenge

# --- Page Configuration ---
st.set_page_config(page_title="NutriVision AI", page_icon="🍎", layout="wide")

# --- Load Data and Models ---
model, class_names = load_model_and_class_names()
NUTRITION_DF = load_nutrition_data()

# --- Initialize Session State for Food Log ---
if 'food_log' not in st.session_state:
    st.session_state.food_log = []
if 'total_intake' not in st.session_state:
    st.session_state.total_intake = {col: 0.0 for col in NUTRITION_DF.columns}
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'daily_goals' not in st.session_state:
    st.session_state.daily_goals = None

# ==============================================================
# ✅ Integrated Alerter Logic (from alerter.py)
# ==============================================================
def generate_alerts(current_intake, daily_goals, new_food_nutrients, food_name):
    """
    Generates contextual nutritional alerts based on adding a new food item.
    Returns a list of dictionaries with the alert 'type' and 'message'.
    """
    alerts = []

    # --- Normalize all nutrient keys to lowercase ---
    current_intake = {k.lower(): v for k, v in current_intake.items()}
    daily_goals = {k.lower(): v for k, v in daily_goals.items()}
    new_food_nutrients = {k.lower(): v for k, v in new_food_nutrients.items()}

    # Calculate projected intake after adding new food
    projected_intake = {
        nutrient: current_intake.get(nutrient, 0) + new_food_nutrients.get(nutrient, 0)
        for nutrient in daily_goals.keys()
    }

    # ✅ Always success message
    alerts.append({
        'type': 'success',
        'message': f"You added **{food_name.title()}** to your daily log! Great choice 👏"
    })

    # 🍭 Sugar Alert
    if 'sugar' in daily_goals:
        sugar_goal = daily_goals.get('sugar', 50)
        projected_sugar = projected_intake.get('sugar', 0)
        sugar_percent = (projected_sugar / sugar_goal) * 100 if sugar_goal else 0

        # Debug check (you can comment this out later)
        print("DEBUG Sugar:", projected_sugar, "/", sugar_goal)

        if projected_sugar > sugar_goal:
            alerts.append({
                'type': 'danger',
                'message': f"🚨 **High Sugar Alert!** Adding {food_name.title()} pushes your sugar intake to {sugar_percent:.1f}% of your daily limit."
            })
        elif projected_sugar > sugar_goal * 0.8:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ **Caution:** You’ve reached {sugar_percent:.1f}% of your sugar goal. Consider reducing sugary items."
            })
        elif projected_sugar > sugar_goal * 0.5:
            alerts.append({
                'type': 'info',
                'message': f"🟡 You’ve reached {sugar_percent:.1f}% of your sugar limit. Keep an eye on your intake."
            })

    # 🧈 Fat Alert
    if 'fat' in daily_goals:
        fat_goal = daily_goals['fat']
        projected_fat = projected_intake.get('fat', 0)
        fat_percent = (projected_fat / fat_goal) * 100 if fat_goal else 0

        if projected_fat > fat_goal:
            alerts.append({
                'type': 'danger',
                'message': f"🚨 **High Fat Alert!** Adding {food_name.title()} exceeds your daily fat goal ({fat_percent:.1f}%)."
            })
        elif projected_fat > fat_goal * 0.8:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ **Warning:** You’ve reached {fat_percent:.1f}% of your daily fat limit."
            })
        elif projected_fat > fat_goal * 0.5:
            alerts.append({
                'type': 'info',
                'message': f"🟡 You’ve reached {fat_percent:.1f}% of your fat limit. Keep your balance healthy."
            })

    # 💪 Protein Encouragement
    if new_food_nutrients.get('protein', 0) > 10:
        alerts.append({
            'type': 'info',
            'message': f"💪 Great Choice! {food_name.title()} is a strong source of protein!"
        })

    return alerts

# ==============================================================


# --- Sidebar ---
with st.sidebar:
    st.title("🍎 NutriVision AI")
    st.write("Your personal nutrition assistant. Upload an image to log your food.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("---")

    # --- User Profile Section ---
    st.header("Your Profile")
    with st.form(key='profile_form'):
        age = st.number_input("Age", 18, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 40, 200, 70)
        activity_level = st.selectbox("Activity Level", ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'])
        submit_button = st.form_submit_button(label='Update Goals')

        if submit_button:
            st.session_state.user_profile = {
                'age': age,
                'gender': gender,
                'weight': weight,
                'activity_level': activity_level
            }
            st.session_state.daily_goals = predict_daily_goals(st.session_state.user_profile)
            st.success("Goals Updated!")

    st.markdown("---")
    st.info("This app uses a machine learning model to identify 36 different types of fruits and vegetables.")

# --- Main Page ---
st.header("Image Analysis")

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    with col2:
        with st.spinner('Analyzing the image...'):
            top_predictions = predict(image, model, class_names)
            best_prediction = top_predictions[0]
            pred_class, confidence = best_prediction

            nutrients = NUTRITION_DF.loc[pred_class].to_dict() if pred_class in NUTRITION_DF.index else {}

            st.subheader("Analysis Result:")
            st.success(f"**{pred_class.replace('_', ' ').title()}** ({confidence:.2f}%)")

            if nutrients:
                st.write("Nutritional Info (per 100g):")
                st.dataframe(pd.DataFrame(nutrients, index=['Value']))

                # --- Generate and Display Alerts ---
                if st.session_state.daily_goals:
                    alerts = generate_alerts(st.session_state.total_intake, st.session_state.daily_goals, nutrients, pred_class)
                    for alert in alerts:
                        if alert['type'] == 'danger':
                            st.error(alert['message'])
                        elif alert['type'] == 'warning':
                            st.warning(alert['message'])
                        elif alert['type'] == 'info':
                            st.info(alert['message'])
                        else:
                            st.success(alert['message'])

                if st.button(f"Add {pred_class.title()} to Daily Log"):
                    st.session_state.last_added_food = pred_class.title()
                    st.session_state.food_log.append(pred_class)
                    for key in st.session_state.total_intake:
                        st.session_state.total_intake[key] += nutrients.get(key, 0)
                    st.rerun()
            else:
                st.warning("Nutritional information not available for this item.")

# --- Display Food Log and Recommendations ---
st.markdown("---")
main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Daily Log & Recommendations", "📈 Trends & Analytics", "🏆 Achievements & Challenges"])

with main_tab1:
    log_col, rec_col = st.columns(2)

    with log_col:
        st.subheader("Your Daily Progress")
        if 'last_added_food' in st.session_state and st.session_state.last_added_food:
            st.success(f"Successfully added {st.session_state.last_added_food} to your log!")
            st.session_state.last_added_food = None

        if st.session_state.food_log:
            st.write(f"**Today's Items:** {', '.join([food.title() for food in st.session_state.food_log])}")
            if st.session_state.daily_goals:
                st.write("**Progress Towards Your Personalized Goals:**")
                progress_df = pd.DataFrame([st.session_state.total_intake, st.session_state.daily_goals], index=['Current', 'Goal'])
                st.dataframe(progress_df)
            else:
                st.info("Update your profile in the sidebar to see personalized goals.")
        else:
            st.info("Your log is empty. Upload an image to add a food item.")

    with rec_col:
        st.subheader("Personalized Recommendations")
        tab1, tab2, tab3, tab4 = st.tabs(["Single Food", "Meal Combos", "Recipe Ideas", "Healthier Substitutes"])

        if not st.session_state.daily_goals:
            st.warning("Please update your profile in the sidebar to get recommendations.")
        else:
            with tab1:
                st.write("##### Need a single item?")
                st.write("Based on your needs, you might like one of these:")
                recommendations = content_based_recommender(
                    st.session_state.total_intake, st.session_state.daily_goals, NUTRITION_DF, st.session_state.food_log
                )
                if recommendations is not None and not recommendations.empty:
                    st.dataframe(recommendations)
                else:
                    st.success("Your nutritional goals are met! No single items needed.")

            with tab2:
                st.write("##### Looking for a meal?")
                st.write("Set your priorities and let our AI find the best combinations for you.")
                priority_col1, priority_col2 = st.columns(2)
                with priority_col1:
                    protein_priority = st.slider("Protein Priority", 1, 5, 3)
                with priority_col2:
                    vit_c_priority = st.slider("Vitamin C Priority", 1, 5, 1)

                priorities = {'protein': protein_priority, 'vitamin_c': vit_c_priority}
                meal_combos = suggest_meal_combinations(
                    st.session_state.total_intake, st.session_state.daily_goals, NUTRITION_DF, st.session_state.food_log, priorities
                )
                if meal_combos:
                    for score, combo in meal_combos:
                        st.success(f"**Combination: {', '.join([c.title() for c in combo])}** (Fitness Score: {score:.4f})")
                        st.dataframe(NUTRITION_DF.loc[list(combo)])
                else:
                    st.success("Your nutritional goals are met! No meal combos needed.")

            with tab3:
                st.write("##### Smart Recipe Suggestions")
                if st.session_state.food_log:
                    last_food = st.session_state.food_log[-1]
                    st.write(f"Here are some recipe ideas that include **{last_food.title()}** and match your nutritional needs:")
                    recipe_suggestions = recommend_recipes(
                        last_food, st.session_state.total_intake, st.session_state.daily_goals, NUTRITION_DF
                    )
                    if recipe_suggestions is not None and not recipe_suggestions.empty:
                        for name, data in recipe_suggestions.iterrows():
                            st.success(f"**{name}**")
                            st.write(f"**Ingredients:** {data['ingredients_list']}")
                            st.write(f"**Link:** {data['url']}")
                    else:
                        st.info(f"No recipes found in our database for '{last_food.title()}' that match your current needs.")
                else:
                    st.info("Add a food to your log to get recipe ideas.")
            
            with tab4:
                st.write("##### Find a Healthier Alternative")
                if st.session_state.food_log:
                    last_food = st.session_state.food_log[-1]
                    st.write(f"Looking for a healthier alternative to **{last_food.title()}**? Here are some ideas with lower calories and sugar:")
                    
                    substitutes = recommend_substitutes(last_food, NUTRITION_DF)

                    if substitutes is not None and not substitutes.empty:
                        st.success(f"Found {len(substitutes)} healthier options:")
                        st.dataframe(substitutes[['calories', 'sugar', 'protein', 'fiber']])
                    else:
                        st.info(f"No direct healthier substitutes found for '{last_food.title()}' based on our criteria.")
                else:
                    st.info("Add a food to your log to get substitute ideas.")

with main_tab2:
    st.header("Your Eating Patterns")
    if st.session_state.food_log:
        # Basic Analytics
        fig_top_foods, fig_cal_trend = get_food_log_analytics(st.session_state.food_log, NUTRITION_DF)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_top_foods, use_container_width=True)
        with col2:
            st.plotly_chart(fig_cal_trend, use_container_width=True)

        st.markdown("---")

        # ML-Powered Diet Diversity Insights
        st.subheader("AI-Powered Diet Analysis")
        insight, cluster_df = get_diet_diversity_insights(st.session_state.food_log, NUTRITION_DF)
        
        st.info(insight)
        if cluster_df is not None:
            st.write("Here's how your logged foods are grouped by nutritional similarity:")
            st.dataframe(cluster_df.sort_values('cluster'))
    else:
        st.info("Log some food items to see your trends and analytics.")

with main_tab3:
    st.header("Your Achievements")
    if st.session_state.food_log and st.session_state.daily_goals:
        # --- ML-Powered Personalized Challenge ---
        st.subheader("Your Personalized Challenge")
        challenge_text, challenge_food, progress = generate_personalized_challenge(st.session_state.food_log, NUTRITION_DF)
        
        # Check if challenge is completed
        if challenge_food and challenge_food in st.session_state.food_log:
            st.balloons()
            st.success(f"🎉 **Challenge Complete!** You ate a {challenge_food.title()}! A new challenge will appear tomorrow.")
        else:
            st.info(challenge_text)
            st.progress(int(progress))
            st.write(f"Dietary Diversity Progress: {int(progress)}%")

        st.markdown("---")

        # --- Badge System ---
        st.subheader("Badges Earned Today")
        earned_badges = check_badges(st.session_state.food_log, st.session_state.total_intake, st.session_state.daily_goals, NUTRITION_DF)
        if earned_badges:
            for badge in earned_badges:
                st.success(f"**{badge['icon']} {badge['name']}** - {badge['description']}")
        else:
            st.info("Log more food items to start earning badges!")
    else:
        st.info("Log food and set your profile goals to unlock achievements and challenges.")
