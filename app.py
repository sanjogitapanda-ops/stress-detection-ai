import streamlit as st
import numpy as np
import joblib
import pandas as pd

# 1. Model load karein (EOFError se bachne ke liye try-except)
try:
    model = joblib.load("stress_model.pkl")
except:
    st.error("⚠️ Model file corrupt hai. Please GitHub par dobara upload karein.")

st.title("🧠 Stress Detection AI System")
st.write("Sahi options select karein aur 'Predict' par click karein.")

# 2. Encoding Mappings (Alphabetical Order - LabelEncoder ke hisaab se)
gender_map = {"Female": 0, "Male": 1}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
yes_no_map = {"No": 0, "Yes": 1} # Smoking aur Meditation ke liye
exercise_map = {
    "Aerobics": 0, "Cardio": 1, "Pilates": 2, 
    "Strength Training": 3, "Walking": 4, "Yoga": 5
}

# Occupation list (Alphabetical order mein sorted)
occupations = [
    "Accountant", "Architect", "Artist", "Business Analyst", "Business Owner", 
    "Chef", "Civil Engineer", "Data Analyst", "Data Scientist", "Doctor", 
    "Electrician", "Engineer", "Freelancer", "Graphic Designer", "HR Manager", 
    "Journalist", "Lawyer", "Marketing Manager", "Nurse", "Pharmacist", 
    "Photographer", "Physician", "Project Manager", "Scientist", "Software Developer", 
    "Software Engineer", "Teacher", "Truck Driver", "Web Developer", "Writer"
]

# 3. Input Form
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", list(gender_map.keys()))
        occupation = st.selectbox("Occupation", occupations)
        marital = st.selectbox("Marital Status", list(marital_map.keys()))
        sleep_dur = st.number_input("Sleep Duration (Hours)", value=7.0)
        sleep_qual = st.slider("Sleep Quality (1-5)", 1, 5, 4)
        wake_time = st.number_input("Wake Up Time (e.g., 7.0 for 7 AM)", value=7.0)
        bed_time = st.number_input("Bed Time (e.g., 22.0 for 10 PM)", value=22.0)
        activity = st.number_input("Physical Activity (Hours)", value=2.0)
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
        chol = st.number_input("Cholesterol Level", value=180.0)
        sugar = st.number_input("Blood Sugar Level", value=90.0)

    submit = st.form_submit_button("Predict Stress Level")

# 4. Prediction Logic
if submit:
    try:
        # User selection ko numbers mein convert karein
        feature_values = [
            age, 
            gender_map[gender], 
            occupations.index(occupation), 
            marital_map[marital],
            sleep_dur, sleep_qual, wake_time, bed_time, activity, screen,
            caffeine, alcohol, yes_no_map[smoking], work_hrs, travel, 
            social, yes_no_map[meditation], exercise_map[exercise],
            bp, chol, sugar
        ]
        
        # Predict karein
        input_array = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(input_array)

        # Result dikhayein (0 = High, 1 = Low/Medium based on your Colab code)
        if prediction == 0:
            st.error("⚠️ HIGH STRESS DETECTED")
        else:
            st.success("✅ LOW / MEDIUM STRESS")
            
    except Exception as e:
        st.error(f"Error: {e}")
