import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Patient Readmission Risk Prediction")

st.write("Predict whether a patient will be readmitted within 30 days.")

# Example inputs (keep it simple)
num_procedures = st.number_input("Number of procedures", min_value=0, max_value=20)
num_medications = st.number_input("Number of medications", min_value=0, max_value=50)
time_in_hospital = st.number_input("Time in hospital (days)", min_value=1, max_value=30)

if st.button("Predict"):
    input_df = pd.DataFrame(
        [[num_procedures, num_medications, time_in_hospital]],
        columns=["num_procedures", "num_medications", "time_in_hospital"]
    )

    # Align with training columns
    full_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    full_input[input_df.columns] = input_df

    full_input[full_input.columns] = scaler.transform(full_input)

    prob = model.predict_proba(full_input)[0][1]

    st.metric("Readmission Risk", f"{prob:.2%}")
