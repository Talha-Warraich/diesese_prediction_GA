# import pandas as pd
# import numpy as np
# import random
# from sklearn.preprocessing import LabelEncoder
## LoadDataset
# file_path = 'Training.csv'
# data = pd.read_csv(file_path)

# # Encode target variable (prognosis)
# le = LabelEncoder()
# data['prognosis'] = le.fit_transform(data['prognosis'])
# disease_names = le.classes_

# # Features and target
# symptom_columns = data.columns[:-1]
# X = data[symptom_columns]
# y = data['prognosis']

# # Symptom list for reference
# symptom_columns = X.columns.tolist()

# def fitness_function(individual, symptom_vector):
#     """Calculate fitness score: match score - penalty for excess symptoms."""
#     match_score = np.sum(individual * symptom_vector)
#     total_symptoms = np.sum(individual)
#     penalty = abs(total_symptoms - np.sum(symptom_vector)) * 0.3
#     return match_score - penalty

# def genetic_algorithm(symptom_vector, population_size=100, generations=100, mutation_rate=0.15):
#     """Run the genetic algorithm to predict the disease."""
#     num_symptoms = len(symptom_vector)
#     population = np.random.randint(2, size=(population_size, num_symptoms))
#     best_solution = None
#     best_fitness = -float('inf')

#     for generation in range(generations):
#         # Evaluate fitness for each individual
#         fitness_scores = np.array([fitness_function(ind, symptom_vector) for ind in population])

#         # Select the top 50% of individuals
#         top_indices = np.argsort(fitness_scores)[-population_size//2:]
#         parents = population[top_indices]

#         # Crossover and mutation to generate offspring
#         offspring = []
#         for _ in range(population_size // 2):
#             p1, p2 = random.sample(list(parents), 2)
#             crossover_point = random.randint(1, num_symptoms - 1)
#             child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
#             offspring.append(child)

#         # Apply mutation
#         for child in offspring:
#             if random.random() < mutation_rate:
#                 mutation_point = random.randint(0, num_symptoms - 1)
#                 child[mutation_point] = 1 - child[mutation_point]

#         # Update population with parents and offspring
#         population = np.vstack((parents, offspring))
#         max_fitness = fitness_scores.max()

#         # Track the best solution
#         if max_fitness > best_fitness:
#             best_fitness = max_fitness
#             best_solution = population[np.argmax(fitness_scores)]

#     return best_solution

# def predict_disease_with_genetic_algorithm(symptoms_present):
#     """Predict the disease and identify uncertainty."""
#     symptom_vector = np.zeros(len(symptom_columns))
#     for symptom in symptoms_present:
#         if symptom in symptom_columns:
#             symptom_vector[symptom_columns.index(symptom)] = 1

#     # Run the genetic algorithm
#     best_solution = genetic_algorithm(symptom_vector)
#     disease_scores = np.dot(X.values, best_solution)  # Score for each disease
#     ranked_diseases = [(disease_names[i], disease_scores[i]) for i in range(len(disease_names))]
#     ranked_diseases = sorted(ranked_diseases, key=lambda x: -x[1])

#     # Check for uncertainty
#     if ranked_diseases[0][1] - ranked_diseases[1][1] < 0.3:
#         return {
#             "prediction": None,
#             "uncertain_diseases": [ranked_diseases[0][0], ranked_diseases[1][0]],
#             "questions": generate_dynamic_questions(ranked_diseases[:2], best_solution)
#         }
#     else:
#         return {"prediction": ranked_diseases[0][0], "questions": None}

# def generate_dynamic_questions(diseases, best_solution):
#     """Generate clarifying questions dynamically based on uncertain diseases."""
#     symptom_differences = {}
#     for disease in diseases:
#         disease_index = np.where(disease_names == disease[0])[0][0]
#         symptom_weights = best_solution * X.iloc[disease_index].values
#         disease_symptoms = [(symptom_columns[i], symptom_weights[i]) for i in range(len(symptom_columns))]
#         disease_symptoms = sorted(disease_symptoms, key=lambda x: -x[1])[:5]
#         symptom_differences[disease[0]] = [symptom[0] for symptom in disease_symptoms]

#     clarifying_questions = []
#     for symptom in set(symptom_differences[diseases[0][0]]) & set(symptom_differences[diseases[1][0]]):
#         clarifying_questions.append(f"Do you have {symptom}? (yes/no)")
#     return clarifying_questions

# def refine_prediction_with_answers(symptoms_present, user_answers):
#     """Refine prediction based on user clarifications."""
#     symptom_vector = np.zeros(len(symptom_columns))
#     for symptom in symptoms_present:
#         if symptom in symptom_columns:
#             symptom_vector[symptom_columns.index(symptom)] = 1

#     # Update symptom vector with user answers
#     for symptom, answer in user_answers.items():
#         if symptom in symptom_columns:
#             symptom_vector[symptom_columns.index(symptom)] = 1 if answer.lower() == "yes" else 0

#     # Calculate scores for each disease

#     disease_scores = []
#     for i in range(len(y)):  # Iterate over all diseases in the dataset
#         individual = X.iloc[i].values
#         disease_scores.append(fitness_function(individual, symptom_vector))

#     # Find the disease with the highest score
#     predicted_disease_index = np.argmax(disease_scores)
#     return disease_names[y.iloc[predicted_disease_index]]

# # Main user interaction loop
# print("Enter symptoms separated by commas (e.g., fever, headache):")
# user_input = input().strip()
# user_symptoms = [symptom.strip().lower() for symptom in user_input.split(",")]

# result = predict_disease_with_genetic_algorithm(user_symptoms)
# if result["questions"] is None:
#     print(f"Predicted disease: {result['prediction']}")
# else:
#     # Handle uncertain diseases
#     print(f"Uncertain diseases: {result['uncertain_diseases']}")
#     print("Let's clarify further with some questions.")
    
#     # Collect user answers for the dynamically generated questions
#     user_answers = {}
#     for question in result["questions"]:
#         symptom = question.split("Do you have ")[1].split("?")[0]
#         user_response = input(f"{question} ").strip().lower()
#         user_answers[symptom] = user_response
    
#     # Refine prediction based on user answers
#     final_disease = refine_prediction_with_answers(user_symptoms, user_answers)
#     print(f"After clarification, the predicted disease is: {final_disease}")



#correct code with randomforest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
file_path = 'Training.csv'
data = pd.read_csv(file_path)

# Encode target variable (prognosis)
le = LabelEncoder()
data['prognosis'] = le.fit_transform(data['prognosis'])
disease_names = le.classes_

# Features and target
X = data.iloc[:, :-1]  # All symptom columns
y = data['prognosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = rf_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Symptom list
symptom_columns = X.columns.tolist()

def predict_disease(symptoms_present):
    """Predict the disease based on symptoms."""
    symptom_vector = np.zeros(len(symptom_columns))
    for symptom in symptoms_present:
        if symptom in symptom_columns:
            symptom_vector[symptom_columns.index(symptom)] = 1

    # Convert to DataFrame with correct column names
    symptom_vector_df = pd.DataFrame([symptom_vector], columns=symptom_columns)
    probabilities = rf_model.predict_proba(symptom_vector_df)[0]
    ranked_probs = sorted([(disease_names[i], prob) for i, prob in enumerate(probabilities)], 
                          key=lambda x: -x[1])

    # Check for uncertainty (close probabilities)
    if ranked_probs[0][1] - ranked_probs[1][1] < 0.3:
        return {
            "prediction": None,
            "uncertain_diseases": [ranked_probs[0][0], ranked_probs[1][0]],
            "questions": generate_questions(ranked_probs[:2])  # Generate questions for top 2 diseases
        }
    else:
        return {"prediction": ranked_probs[0][0], "questions": None}

def generate_questions(diseases):
    """Generate clarifying questions based on similar diseases."""
    symptom_differences = {}
    for disease in diseases:
        # Find symptoms most strongly associated with each disease
        disease_index = np.where(disease_names == disease[0])[0][0]
        symptom_weights = rf_model.feature_importances_
        disease_symptoms = [(symptom_columns[i], symptom_weights[i]) for i in range(len(symptom_columns))]
        disease_symptoms = sorted(disease_symptoms, key=lambda x: -x[1])[:5]
        symptom_differences[disease[0]] = [symptom[0] for symptom in disease_symptoms]
    
    # Compare top symptoms of uncertain diseases
    clarifying_questions = []
    for symptom in set(symptom_differences[diseases[0][0]]) & set(symptom_differences[diseases[1][0]]):
        clarifying_questions.append(f"Do you have {symptom}? (yes/no)")
    return clarifying_questions

def refine_prediction(user_symptoms, user_answers):
    """Refine prediction based on clarifications."""
    symptom_vector = np.zeros(len(symptom_columns))
    for symptom in user_symptoms:
        if symptom in symptom_columns:
            symptom_vector[symptom_columns.index(symptom)] = 1

    # Update symptom vector with user answers
    for symptom, answer in user_answers.items():
        if symptom in symptom_columns:
            symptom_vector[symptom_columns.index(symptom)] = 1 if answer.lower() == "yes" else 0

    prediction = rf_model.predict([symptom_vector])[0]
    return disease_names[prediction]

# Main user interaction loop
print("Enter symptoms separated by commas (e.g., fever, headache):")
user_input = input()
user_symptoms = [symptom.strip().lower() for symptom in user_input.split(",")]

result = predict_disease(user_symptoms)

if result["questions"] is None:
    print(f"Predicted disease: {result['prediction']}")
else:
    print(f"Uncertain diseases: {result['uncertain_diseases']}")
    user_answers = {}
    for question in result["questions"]:
        symptom = question.split("Do you have ")[1].split("?")[0]
        user_answers[symptom] = input(f"{question} ").strip().lower()
    final_disease = refine_prediction(user_symptoms, user_answers)
    print(f"Final predicted disease: {final_disease}")
