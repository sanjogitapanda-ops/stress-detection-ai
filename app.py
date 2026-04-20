import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Model load karein
try:
    model = joblib.load("stress_model.pkl")
except Exception as e:
    st.error(f"Model load karne mein error: {e}")

st.title("🧠 Stress Detection AI System")

# Dataset load karein column names ke liye
df = pd.read_csv("stress.csv")
features = df.drop("Stress_Detection", axis=1).columns

with st.form("input_form"):
    inputs = []
    for col in features:
        # 0.00 hatane ke liye text_input use kiya hai
        val = st.text_input(f"Enter {col}", placeholder="Type value here...")
        inputs.append(val)
    
    submit = st.form_submit_button("Predict Stress")

if submit:
    try:
        # Inputs ko number mein badlein
        input_array = np.array([float(i) for i in inputs]).reshape(1, -1)
        prediction = model.predict(input_array)

        if prediction == 1:
            st.error("⚠ HIGH STRESS DETECTED")
        else:
            st.success("✅ LOW STRESS")
    except ValueError:
        st.warning("Please enter numbers only in all fields.")
