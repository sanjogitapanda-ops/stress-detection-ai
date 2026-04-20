import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Model load karein
model = joblib.load("stress_model.pkl")

st.title("🧠 Stress Detection AI System")

# Dataset load karein columns aur options ke liye
df = pd.read_csv("stress.csv")
features = df.drop("Stress_Detection", axis=1).columns

# Sabhi unique occupations ki list banayein [1, 2]
occupation_list = sorted(df['Occupation'].unique().tolist())
occupation_list.append("Other")

with st.form("user_form"):
    inputs_dict = {}
    
    for col in features:
        if col == "Occupation":
            # Dropdown menu banayein
            selected_occ = st.selectbox("Select Occupation", occupation_list)
            
            # Agar "Other" select kiya toh text box dikhayein
            if selected_occ == "Other":
                other_occ = st.text_input("Please type your occupation", placeholder="Type here...")
                # Model ke liye hum default value 'Other' ya pehli occupation use karenge
                inputs_dict[col] = "Other" 
            else:
                inputs_dict[col] = selected_occ
        
        elif col == "Gender":
            inputs_dict[col] = st.selectbox("Gender", ["Male", "Female"])
        
        else:
            inputs_dict[col] = st.text_input(f"Enter {col}", placeholder="Type value...")

    submit = st.form_submit_button("Predict Stress")

if submit:
    try:
        # Data ko encode karna zaroori hai kyunki model numbers mangta hai
        processed_inputs = []
        for col in features:
            val = inputs_dict[col]
            
            # Agar column text hai toh use number mein badlein (Label Encoding)
            if df[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                
                # Agar unknown occupation hai toh use 'Other' ki tarah treat karein
                try:
                    encoded_val = le.transform([str(val)])
                except:
                    encoded_val = 0 # Default value agar model ne woh naam na suna ho
                processed_inputs.append(encoded_val)
            else:
                processed_inputs.append(float(val))

        # Prediction dikhayein
        prediction = model.predict(np.array(processed_inputs).reshape(1, -1))
        if prediction == 1:
            st.error("⚠ HIGH STRESS DETECTED")
        else:
            st.success("✅ LOW STRESS")
            
    except Exception as e:
        st.error(f"Error: {e}. Please ensure all fields are filled correctly.")
