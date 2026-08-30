import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("student_exam_score_model.pkl")
preprocessor = joblib.load("student_exam_score_preprocessor.pkl")

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Enter the student's academic and personal factors "
    "to predict their exam score."
)

st.divider()

# Numeric inputs
hours_studied = st.number_input(
    "Hours Studied",
    min_value=0,
    max_value=24,
    value=5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0,
    max_value=24,
    value=7
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=70
)

tutoring_sessions = st.number_input(
    "Tutoring Sessions",
    min_value=0,
    max_value=20,
    value=2
)

physical_activity = st.number_input(
    "Physical Activity (hours/week)",
    min_value=0,
    max_value=20,
    value=3
)

# Categorical inputs
parental_involvement = st.selectbox(
    "Parental Involvement",
    ["Low", "Medium", "High"]
)

access_to_resources = st.selectbox(
    "Access to Resources",
    ["Low", "Medium", "High"]
)

extracurricular_activities = st.selectbox(
    "Extracurricular Activities",
    ["No", "Yes"]
)

motivation_level = st.selectbox(
    "Motivation Level",
    ["Low", "Medium", "High"]
)

internet_access = st.selectbox(
    "Internet Access",
    ["No", "Yes"]
)

family_income = st.selectbox(
    "Family Income",
    ["Low", "Medium", "High"]
)

teacher_quality = st.selectbox(
    "Teacher Quality",
    ["Low", "Medium", "High"]
)

school_type = st.selectbox(
    "School Type",
    ["Public", "Private"]
)

peer_influence = st.selectbox(
    "Peer Influence",
    ["Negative", "Neutral", "Positive"]
)

learning_disabilities = st.selectbox(
    "Learning Disabilities",
    ["No", "Yes"]
)

parental_education_level = st.selectbox(
    "Parental Education Level",
    ["High School", "College", "Postgraduate"]
)

distance_from_home = st.selectbox(
    "Distance from Home",
    ["Near", "Moderate", "Far"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

st.divider()

if st.button("🔮 Predict Exam Score", use_container_width=True):

    input_data = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular_activities,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education_level,
        "Distance_from_Home": distance_from_home,
        "Gender": gender
    }])

    # Apply the same preprocessing used during training
    processed_data = preprocessor.transform(input_data)

    # Generate prediction
    prediction = model.predict(processed_data)[0]

    st.success(f"🎯 Predicted Exam Score: {prediction:.2f}")

    if prediction >= 80:
        st.balloons()
        st.info("Excellent predicted performance! 🌟")
    elif prediction >= 60:
        st.info("Good predicted performance! 👍")
    else:
        st.warning("The prediction suggests that additional academic support may help. 📚")