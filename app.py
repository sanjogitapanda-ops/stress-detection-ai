import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("stress_model.pkl")

st.title("🧠 Stress Detection AI System")
st.write("Enter the required details below and click 'Predict Stress'.")

# Load the CSV to get the column names automatically
# Ensure stress.csv is uploaded to your GitHub repository
df = pd.read_csv("stress.csv")
target = "Stress_Detection"
features = df.drop(target, axis=1).columns

# Using a form helps group inputs and allows you to use 'Tab' to move between them
with st.form("user_input_form"):
    inputs = []
    
    for col in features:
        # st.text_input stays empty (no 0.00) and has no irritating buttons
        val = st.text_input(f"Enter {col}", placeholder="Type value here...")
        inputs.append(val)
    
    # Submit button for the form
    submit = st.form_submit_button("Predict Stress")

if submit:
    try:
        # Check if any field was left empty
        if any(x == "" for x in inputs):
            st.warning("Please fill in all the fields before predicting.")
        else:
            # Convert text inputs to numbers for the model
            input_array = np.array([float(i) for i in inputs]).reshape(1, -1)
            
            # Make the prediction
            prediction = model.predict(input_array)

            if prediction == 1:
                st.error("⚠ HIGH STRESS DETECTED")
            else:
                st.success("✅ LOW STRESS")
                
    except ValueError:
        st.warning("Please enter valid numbers in all fields (e.g., use 5.5 instead of text).")
