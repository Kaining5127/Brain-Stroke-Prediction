from flask import Flask, request, render_template
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Load the trained SVM model
with open('svm_model_with_optimal_features.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the expected feature names for the model
model_features = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        data = {
            "gender": request.form['gender'],
            "age": float(request.form['age']),
            "hypertension": int(request.form['hypertension']),
            "heart_disease": int(request.form['heart_disease']),
            "ever_married": request.form['ever_married'],
            "work_type": request.form['work_type'],
            "Residence_type": request.form['Residence_type'],
            "avg_glucose_level": float(request.form['avg_glucose_level']),
            "bmi": float(request.form['bmi']),
        }

        # Convert input data into a DataFrame
        input_df = pd.DataFrame([data])

        # Preprocessing consistent with training
        input_df['bmi'] = input_df['bmi'].fillna(input_df['bmi'].mean())

        # Map categorical columns
        input_df['gender'] = input_df['gender'].map({'Male': 0, 'Female': 1})
        input_df['ever_married'] = input_df['ever_married'].map({'No': 0, 'Yes': 1})
        input_df['Residence_type'] = input_df['Residence_type'].map({'Rural': 0, 'Urban': 1})

        # One-hot encoding for 'work_type'
        input_df = pd.get_dummies(input_df, columns=['work_type'], drop_first=True)

        # Ensure that all model features are present in the input data
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Predict using the trained model
        prediction = model.predict(input_df)

        # Return prediction result in a webpage
        return render_template('index.html', prediction=int(prediction[0]))

    except Exception as e:
        # Handle errors and pass the error message to the template
        return render_template('index.html', error="An error occurred: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)


