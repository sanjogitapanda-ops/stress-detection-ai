import streamlit as st
import numpy as np
import joblib
import pandas as pd

# 1. Model load karein
try:
    model = joblib.load("stress_model.pkl")
except:
    st.error("⚠️ Model file load nahi ho rahi. Please check karein ki 'stress_model.pkl' GitHub par sahi se upload hui hai.")

st.title("🧠 Stress Detection AI System")
st.write("Apni details niche bharein. Agar aapka option list mein nahi hai, toh 'Other' select karein.")

# 2. Categorical Mapping (Alphabetical order based on LabelEncoder)
gender_map = {"Female": 0, "Male": 1}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
yes_no_map = {"No": 0, "Yes": 1}

# Exercise Type mapping with 'No Exercise/Other'
exercise_map = {
    "Aerobics": 0, 
    "Cardio": 1, 
    "Pilates": 2, 
    "Strength Training": 3, 
    "Walking": 4, 
    "Yoga": 5,
    "No Exercise / Other": 4  # Defaulting to 'Walking' index for neutral prediction
}

# Occupation List with 'Other'
occupations = [
    "Accountant", "Architect", "Artist", "Business Analyst", "Business Owner", 
    "Chef", "Civil Engineer", "Consultant", "Data Scientist", "Doctor", 
    "Electrician", "Engineer", "Freelancer", "Graphic Designer", "HR Manager", 
    "Journalist", "Lawyer", "Manager", "Marketing Manager", "Nurse", 
    "Pharmacist", "Photographer", "Physician", "Project Manager", "Scientist", 
    "Software Engineer", "Teacher", "Truck Driver", "Web Developer", "Writer", "Other"
]

# 3. Input Form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("Gender", list(gender_map.keys()))
        occ_choice = st.selectbox("Occupation", occupations)
        # Agar "Other" select ho toh extra box dikhane ke liye hum form ke bahar logic nahi laga sakte 
        # isliye Other ko hum Accountant (0) ki tarah treat karenge prediction ke liye.
        marital = st.selectbox("Marital Status", list(marital_map.keys()))
        sleep_dur = st.number_input("Sleep Duration (Hours)", value=7.0)
        sleep_qual = st.slider("Sleep Quality (1-5)", 1, 5, 4)
        wake_time = st.number_input("Wake Up Time (e.g., 7.0 for 7 AM)", value=7.0)
        bed_time = st.number_input("Bed Time (e.g., 22.0 for 10 PM)", value=22.0)
        activity = st.number_input("Physical Activity (Hours)", value=1.0)
        screen = st.number_input("Screen Time (Hours)", value=4.0)

    with col2:
        caffeine = st.number_input("Caffeine Intake (Cups)", value=1.0)
        alcohol = st.number_input("Alcohol Intake (Units)", value=0.0)
        smoking = st.selectbox("Smoking Habit", list(yes_no_map.keys()))
        work_hrs = st.number_input("Work Hours", value=8.0)
        travel = st.number_input("Travel Time (Hours)", value=1.0)
        social = st.number_input("Social Interactions", value=5.0)
        meditation = st.selectbox("Meditation Practice", list(yes_no_map.keys()))
        exercise = st.selectbox("Exercise Type", list(exercise_map.keys()))
        bp = st.number_input("Blood Pressure", value=120.0)
        sugar = st.number_input("Blood Sugar Level", value=90.0)

    submit = st.form_submit_button("Predict Stress Level")

# 4. Prediction Logic
if submit:
    try:
        # Occupation index handle karein
        if occ_choice == "Other":
            occ_val = 0 # Defaulting to first index
        else:
            occ_val = occupations.index(occ_choice)

        feature_values = [
            age, gender_map[gender], occ_val, marital_map[marital],
            sleep_dur, sleep_qual, wake_time, bed_time, activity, screen,
            caffeine, alcohol, yes_no_map[smoking], work_hrs, travel, 
            social, yes_no_map[meditation], exercise_map[exercise],
            bp,sugar
        ]
        
        prediction = model.predict(np.array(feature_values).reshape(1, -1))

        if prediction == 0:
            st.error("⚠️ HIGH STRESS DETECTED")
        else:
            st.success("✅ LOW / MEDIUM STRESS")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
