from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load('sleep_disorder_model.pkl')
scaler = joblib.load('scaler.pkl')

with open('Gender_encoder.pkl', 'rb') as f:
    gender_le = pickle.load(f)
with open('Occupation_encoder.pkl', 'rb') as f:
    occupation_le = pickle.load(f)
with open('BMI_Category_encoder.pkl', 'rb') as f:
    bmi_le = pickle.load(f)
with open('Sleep_Disorder_encoder.pkl', 'rb') as f:
    sleep_le = pickle.load(f)

# Advice messages based on prediction
precautions = {
    'Sleep Apnea': "Avoid alcohol before bedtime, maintain a healthy weight, and consult a sleep specialist if snoring is severe.",
    'Insomnia': "Maintain a regular sleep schedule, avoid caffeine late in the day, limit screen time before bed, and practice relaxation techniques.",
    'None': "Great job! Keep following healthy sleep habits and maintaining a balanced lifestyle to support continued well-being."
}

@app.route('/')
def home():
    return render_template(
        'index.html',
        genders=gender_le.classes_,
        occupations=occupation_le.classes_,
        bmi_categories=bmi_le.classes_,
        prediction=None,
        advice=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from form
        input_data = {
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Occupation': request.form['Occupation'],
            'Sleep Duration': float(request.form['SleepDuration']),
            'Quality of Sleep': float(request.form['QualityOfSleep']),
            'Physical Activity Level': float(request.form['PhysicalActivityLevel']),
            'Stress Level': float(request.form['StressLevel']),
            'BMI Category': request.form['BMICategory'],
            'Heart Rate': int(request.form['HeartRate']),
            'Daily Steps': int(request.form['DailySteps']),
            'Systolic': int(request.form['Systolic']),
            'Diastolic': int(request.form['Diastolic'])
        }

        df = pd.DataFrame([input_data])

        # Encode categorical variables
        df['Gender'] = gender_le.transform(df['Gender'])
        df['Occupation'] = occupation_le.transform(df['Occupation'])
        df['BMI Category'] = bmi_le.transform(df['BMI Category'])

        # Scale numerical features
        numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep',
                          'Physical Activity Level', 'Stress Level',
                          'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        pred_encoded = model.predict(df)[0]
        pred = sleep_le.inverse_transform([pred_encoded])[0]

        result = f"Predicted Sleep Disorder: {pred}"
        advice = precautions.get(pred, "")

    except Exception as e:
        result = f"Error: {e}"
        advice = ""

    return render_template(
        'index.html',
        genders=gender_le.classes_,
        occupations=occupation_le.classes_,
        bmi_categories=bmi_le.classes_,
        prediction=result,
        advice=advice
    )

if __name__ == '__main__':
    app.run(debug=True)
