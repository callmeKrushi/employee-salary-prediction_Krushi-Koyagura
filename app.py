import streamlit as st
import pandas as pd
import joblib
import os

# Load best model
model = joblib.load("best_model.pkl")

# Load encoders
encoder_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'native-country']
encoders = {}
for col in encoder_cols:
    enc_path = os.path.join("encoders", f"{col}_encoder.pkl")
    encoders[col] = joblib.load(enc_path)

# Load feature order
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

st.title("💼 Employee Salary Prediction")
st.write("Enter employee details below to predict if income is >50K or ≤50K.")

# User input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
    education = st.selectbox("Education", encoders['education'].classes_)
    marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
    occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
    relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
    race = st.selectbox("Race", encoders['race'].classes_)
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
    native_country = st.selectbox("Native Country", encoders['native-country'].classes_)

    submit = st.form_submit_button("Predict Salary")

# On submit
if submit:
    input_dict = {
        'age': age,
        'workclass': encoders['workclass'].transform([workclass])[0],
        'education': encoders['education'].transform([education])[0],
        'marital-status': encoders['marital-status'].transform([marital_status])[0],
        'occupation': encoders['occupation'].transform([occupation])[0],
        'relationship': encoders['relationship'].transform([relationship])[0],
        'race': encoders['race'].transform([race])[0],
        'gender': encoders['gender'].transform([gender])[0],
        'hours-per-week': hours_per_week,
        'native-country': encoders['native-country'].transform([native_country])[0]
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_order]  # Ensure column order

    prediction = model.predict(input_df)[0]
    result = ">50K" if prediction == 1 else "≤50K"

    st.success(f"Predicted Income: **{result}**")
st.markdown("---")
st.markdown("<h5 style='color: #e9c46a;'>Made with ❤️ by [KRUSHI KOYAGURA] — A Beginner ML Project</h5>", unsafe_allow_html=True)
