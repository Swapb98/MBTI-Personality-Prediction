import os
import pickle
import tensorflow as tf
from keras.models import load_model

# Define the MBTI dichotomies
combinations = ['EI', 'NS', 'TF', 'JP']

# Load the saved models
l_models = {combination: load_model(f'model/bin_{combination}.h5') for combination in combinations}

# Load the saved vectorizer
with open('model/tfidf.pkl', 'rb') as f:
    l_vectorizer = pickle.load(f)

# Simulate new data (replace this with your actual new data)
new_post = "this is a new post another post for testing yet another post"

# Preprocess new data using the loaded vectorizer
X_new = l_vectorizer.transform([new_post]).toarray()

# Make predictions for each combination
y_preds = []
for combination, model in l_models.items():
    y_pred = model.predict(X_new)
    # For binary classification, prediction is a probability
    if y_pred >= 0.5:
        y_preds.append(combination[1])  # second character
    else:
        y_preds.append(combination[0])  # first character

# Concatenate the predictions for each combination into a single string
prediction = ''.join(y_preds)

# Print the concatenated prediction
print(f"Prediction: {prediction}")
