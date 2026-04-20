
import streamlit as st
import numpy as np
import joblib

model = joblib.load("stress_model.pkl")

st.title("🧠 Stress Detection AI System")

st.write("Enter values for prediction")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")

if st.button("Predict"):
    data = np.array([[f1, f2, f3, f4]])
    pred = model.predict(data)[0]

    if pred == 1:
        st.error("HIGH STRESS ⚠")
    else:
        st.success("LOW STRESS ✅")
