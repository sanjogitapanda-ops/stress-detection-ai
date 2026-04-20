import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("stress_model.pkl")

st.title("🧠 Stress Detection AI System")

st.write("Enter values based on dataset features")

# Load dataset just to get feature structure (IMPORTANT FIX)
df = pd.read_csv("stress.csv")
target = "Stress_Detection"

features = df.drop(target, axis=1).columns

inputs = []

# Dynamically create inputs
for col in features:
    val = st.number_input(f"{col}", value=0.0)
    inputs.append(val)

# Predict
if st.button("Predict Stress"):
    input_array = np.array(inputs).reshape(1, -1)
    
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("⚠ HIGH STRESS DETECTED")
    else:
        st.success("✅ LOW STRESS")
