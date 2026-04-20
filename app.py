import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("stress_model.pkl")

st.title("🧠 Stress Detection AI System")
st.write("Enter values below and press 'Predict'")

# Load dataset for feature names
df = pd.read_csv("stress.csv")
target = "Stress_Detection"
features = df.drop(target, axis=1).columns

# Use a form to prevent the page from refreshing after every single input
with st.form("input_form"):
    inputs = []
    
    for col in features:
        # st.text_input removes the "0.00" and the +/- buttons
        val = st.text_input(f"Enter {col}", placeholder="Type here...")
        inputs.append(val)
    
    # The submit button for the form
    submit_button = st.form_submit_button("Predict Stress")

if submit_button:
    try:
        # Convert text inputs to numbers for the model
        input_array = np.array([float(i) for i in inputs]).reshape(1, -1)
        
        prediction = model.predict(input_array)

        if prediction == 1:
            st.error("⚠ HIGH STRESS DETECTED")
        else:
            st.success("✅ LOW STRESS")
    except ValueError:
        st.warning("Please enter valid numbers in all fields."
