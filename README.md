# SleepSync - Sleep Disorder Prediction Using Machine Learning 🧠💤

This project helps identify potential sleep disorders such as **Insomnia** and **Sleep Apnea** by analyzing lifestyle and health-related features like sleep duration, stress levels, and physical activity. The model is trained using machine learning techniques and integrated with a Flask web application for easy user interaction.

---

## 📖 About

Sleep disorders affect millions and are often undiagnosed. This project provides a solution by predicting potential disorders based on health and behavioral data. It uses a labeled dataset and supervised ML models to make predictions.

Key Features:
- Data cleaning, visualization, and preprocessing
- Model training with SMOTE balancing
- Flask-based prediction UI
- Encoders and scaler saved for real-time prediction

---

## 🗃️ Dataset Description

The model is trained on the Sleep Health and Lifestyle Dataset, which includes:

- **Number of samples:** 374  
- **Features:** Age, Gender, Occupation, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps, etc.  
- **Target:** Sleep Disorder classification (None, Insomnia, Sleep Apnea)  

Data preprocessing included encoding categorical variables, splitting blood pressure into systolic and diastolic, and balancing classes with SMOTE.

---

## 📊 Data Exploration & Preprocessing Summary

- Checked for missing values; `Sleep Disorder` had 219 missing values filled with "None"
- Encoded categorical features (`Gender`, `Occupation`, `BMI Category`, `Sleep Disorder`) using `LabelEncoder` and saved encoders with `pickle`
- Split `Blood Pressure` into two numerical features: `Systolic` and `Diastolic`
- Scaled numerical features using `StandardScaler`
- Balanced imbalanced classes with SMOTE on training data
- Dropped irrelevant columns such as `Person ID`

---

## 📊 Model Details and Performance

- **Algorithm:** Random Forest Classifier  
- **Training:** On SMOTE balanced data  
- **Test Accuracy:** 96%  
- **Metrics:** Precision, Recall, F1-score for each class (Insomnia, None, Sleep Apnea)  
- Confusion matrix and feature importance visualizations generated  
- Model and scaler saved with `joblib` for deployment

---

## 🧰 Technologies Used
- Python  
- Flask (Web framework)  
- HTML  
- CSS  

## 📚 Libraries and Tools
- scikit-learn, pandas, numpy, matplotlib, seaborn, imblearn (SMOTE)  


## 🖼️ Screenshots

![Input Form](screenshots/input_form.png)  
*Input form for user data*

![Prediction Result](screenshots/prediction_result.png)  
*Prediction result displayed*

