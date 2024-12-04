import streamlit as st
import numpy as np 
import joblib
import os

# Define the local image path
image_path = "doc.jpg"  # Replace with your image filename

# Background Image Styling using Local Image
page_bg_img = f'''
<style>
body {{
    background-image: url("file://{os.path.abspath(image_path)}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white; /* To ensure text is readable */
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and description
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Welcome to My Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>CardioShield: Early Prediction for a Safer Tomorrow</p>", unsafe_allow_html=True)

# Load models
try:
    rf = joblib.load(os.path.join('Random Forest.pkl'))
    svm = joblib.load(os.path.join('Support Vector Machine.pkl'))
except FileNotFoundError:
    rf, svm = None, None  # Set to None if not found

# Prediction function
def predict_heart_disease(input_data, model):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Model selection
model_dict = {
    'Random Forest': rf,
    'SVM': svm
}
model_choice = st.selectbox("Which model do you want to use?", options=list(model_dict.keys()))
selected_model = model_dict[model_choice]

# Input fields
sex_options = [0, 1]
sex_labels = ['Female', 'Male']
sex = st.selectbox(label="Sex", options=sex_options, format_func=lambda x: sex_labels[x])
age = st.slider(label='Enter Your Age:', min_value=1, max_value=100, value=25)

col1, col2 = st.columns(2)
cp_options = [3, 1, 2, 0]
ecg_options = [1, 2, 0]
ecg_labels = ['Normal', 'ST', 'LVH']
cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-Anginal pain', 'Asymptomatic']
with col1:
    cp = st.selectbox(label="Chest Pain", options=cp_options, format_func=lambda x: cp_labels[x])
    rbp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)

with col2:
    restecg = st.selectbox("Resting Electrocardiographic Results (RestECG)", options=ecg_options, format_func=lambda x: ecg_labels[x])
    chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)

slope_options = [2, 0, 1]
slope_labels = ['Up', 'Down', 'Flat']
maxhr = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=202, value=150)
slope = st.selectbox("Slope of Peak Exercise ST Segment (Slope)", options=slope_options, format_func=lambda x: slope_labels[x])
oldpeak = st.number_input("ST Depression Induced by Exercise (Oldpeak)", min_value=0.0, max_value=6.0, step=0.1, value=1.0)

col3, col4 = st.columns(2)
with col3:
    fbs = 1 if st.checkbox("Do You Have Fasting Blood Sugar > 120 mg/dl") else 0
with col4:
    xang = 1 if st.checkbox("Do You Have Exercise-Induced Angina") else 0

# Prediction button
if st.button('Predict'):
    if selected_model is None:
        st.error(f"The selected model ({model_choice}) is not available. Please ensure the model is trained and saved.")
    else:
        input_data = [age, sex, cp, rbp, chol, fbs, restecg, maxhr, xang, oldpeak, slope]
        prediction = predict_heart_disease(input_data, selected_model)

        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("Heart Disease Detected!")
        else:
            st.success("No Heart Disease Detected!")

# Disclaimer
st.write("Disclaimer: This app is for educational purposes only and should not be used as a substitute for professional medical advice.")
