import numpy as np
import pandas as pd
import random

# --- Genetic Algorithm Configuration ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 30
ELITISM_COUNT = 2  # Number of best individuals to carry over to the next generation
MUTATION_RATE = 0.1

def _calculate_fitness(meal, ideal_remaining_vector, nutrient_vectors, priorities):
    """
    Calculates how 'fit' a meal is. A higher score is better.
    The fitness is a weighted score based on how well the meal's nutrients
    match the remaining needs, adjusted by user priorities.
    """
    if not meal:
        return 0

    # Get the combined nutrient vector for the current meal
    meal_vector = nutrient_vectors.loc[list(meal)].sum().values

    # Calculate the difference between what the meal provides and what is needed
    # We penalize both under-providing and over-providing nutrients.
    diff = np.abs(meal_vector - ideal_remaining_vector)

    # Apply priority weights. Higher priority means a larger penalty for deviation.
    weighted_diff = diff * priorities

    # The fitness is inversely related to the total weighted difference.
    # We add 1 to the denominator to avoid division by zero.
    fitness_score = 1 / (np.sum(weighted_diff) + 1)

    return fitness_score

def suggest_meal_combinations(current_intake, daily_goals, nutrition_df, food_log, priorities, combo_size=3, num_results=3):
    """
    Uses a Genetic Algorithm to evolve the best meal combinations.
    """
    # Define the nutrient columns based on the main nutrition dataframe
    nutrient_cols = nutrition_df.columns.tolist()

    # 1. Calculate the "ideal" remaining nutrients needed
    goal_vector = np.array([daily_goals.get(col, 0) for col in nutrient_cols])
    current_vector = np.array([current_intake.get(col, 0) for col in nutrient_cols])
    ideal_remaining_vector = (goal_vector - current_vector).clip(0, None)

    if np.sum(ideal_remaining_vector) < 10:
        return []

    # 2. Prepare data for the GA
    available_foods_df = nutrition_df.loc[~nutrition_df.index.isin(food_log)]
    if len(available_foods_df) < combo_size:
        return []

    food_list = available_foods_df.index.tolist()
    nutrient_vectors = available_foods_df[nutrient_cols]
    priority_vector = np.array([priorities.get(col, 1) for col in nutrient_cols])

    # 3. Initialize Population: Create a set of random meals
    population = [random.sample(food_list, combo_size) for _ in range(POPULATION_SIZE)]

    # 4. Evolution Loop
    for _ in range(NUM_GENERATIONS):
        # --- Fitness Calculation & Sorting ---
        # Pair each meal with its fitness score and sort the population from best to worst
        pop_with_fitness = [
            (_calculate_fitness(meal, ideal_remaining_vector, nutrient_vectors, priority_vector), meal)
            for meal in population
        ]
        pop_with_fitness.sort(key=lambda x: x[0], reverse=True)

        next_generation = []

        # --- Elitism: Carry over the best individuals ---
        elites = [meal for score, meal in pop_with_fitness[:ELITISM_COUNT]]
        next_generation.extend(elites)

        # --- Selection (Tournament Selection) ---
        # Fill the rest of the generation
        for _ in range(POPULATION_SIZE - ELITISM_COUNT):
            # Select two random meals and keep the one with the higher fitness
            parent1_score, parent1_meal = random.choice(pop_with_fitness)
            parent2_score, parent2_meal = random.choice(pop_with_fitness)
            winner = parent1_meal if parent1_score > parent2_score else parent2_meal
            next_generation.append(list(winner)) # Add a copy

        population = next_generation

        # --- Crossover & Mutation ---
        # Apply to the non-elite part of the population
        for i in range(ELITISM_COUNT, POPULATION_SIZE, 2):
            parent1, parent2 = population[i], population[i+1]
            
            # Crossover with Repair: Swap one food item and fix duplicates
            crossover_point = random.randint(0, combo_size - 1)
            
            # Perform the swap
            swapped_item1, swapped_item2 = parent1[crossover_point], parent2[crossover_point]
            parent1[crossover_point], parent2[crossover_point] = swapped_item2, swapped_item1

            # Repair step to eliminate duplicates created by crossover
            if len(set(parent1)) < combo_size:
                for idx, item in enumerate(parent1):
                    if item == swapped_item2 and idx != crossover_point:
                        parent1[idx] = swapped_item1
                        break
            if len(set(parent2)) < combo_size:
                for idx, item in enumerate(parent2):
                    if item == swapped_item1 and idx != crossover_point:
                        parent2[idx] = swapped_item2
                        break

            # Guaranteed Unique Mutation
            if random.random() < MUTATION_RATE:
                possible_new_foods = [food for food in food_list if food not in parent1 and food not in parent2]
                if possible_new_foods:
                    parent1[random.randint(0, combo_size - 1)] = random.choice(possible_new_foods)

    # 5. Final Selection: Get the best meals from the final population
    final_fitness = [(_calculate_fitness(meal, ideal_remaining_vector, nutrient_vectors, priority_vector), meal) for meal in population]
    final_fitness.sort(key=lambda x: x[0], reverse=True)

    # Return unique top results
    unique_results = []
    for score, combo in final_fitness:
        sorted_combo = tuple(sorted(combo))
        if sorted_combo not in [tuple(sorted(c)) for s, c in unique_results]:
            unique_results.append((score, combo))
        if len(unique_results) >= num_results:
            break
            
    return unique_results
