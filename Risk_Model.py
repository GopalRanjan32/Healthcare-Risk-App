
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Risk_model1.pkl')

st.title("Healthcare Risk Stratisfication App")

age = st.number_input("Age", min_value=0)
length_of_stay = st.number_input("Length of Stay (days)", min_value=0)
treatment_cost = st.number_input("Treatment Cost ", min_value=0.0)
AbnormalLabCount = st.number_input("AbnormalLabCount", min_value=0)

if st.button("Predict Risk"):
    input_data = pd.DataFrame({
        'Age': [age],
        'LengthOfStay': [length_of_stay],
        'TreatmentCost': [treatment_cost],
        'AbnormalLabCount': [AbnormalLabCount]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.write(f"Predicted Risk: {'High' if prediction == 1 else 'Low'}")
    st.write(f"Probability of High Risk: {probability:.2f}")