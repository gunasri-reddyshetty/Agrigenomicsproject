from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from Bio import AlignIO
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models and other necessary data
model_classification = pickle.load(open('model1.pkl', 'rb'))
model_regression = pickle.load(open('model2.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the training data to fit LabelEncoder
training_data = pd.read_csv('data.csv')  # Replace 'training_data.csv' with the actual file name
training_labels = training_data['Subpopulation'].unique()

# Instantiate LabelEncoder
le = LabelEncoder()

# Fit LabelEncoder with training labels
le.fit(training_labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sequence = request.form['sequence']

        # Process the sequence
        new_input_kmers = vectorizer.transform([sequence]).toarray()
        new_input_df = pd.DataFrame(new_input_kmers)
        new_input_df_scaled = scaler.transform(new_input_df)

        # Classification prediction
        predicted_class = model_classification.predict(new_input_df_scaled)[0]

        # Regression prediction
        new_subpopulation_encoded = le.transform([predicted_class])
        new_data_encoded = np.concatenate((new_input_df_scaled, new_subpopulation_encoded.reshape(-1, 1)), axis=1)
        plant_height_prediction = model_regression.predict(new_data_encoded)[0]

        return render_template('result.html', sequence=sequence, predicted_class=predicted_class, plant_height=plant_height_prediction)

if __name__ == '__main__':
    app.run(debug=True)
