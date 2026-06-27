import streamlit as st
import pandas as pd
import joblib
import os
import gdown

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_large_model():
    model_path = "diabetes_stacking_model.pkl"
    file_id = "1GRVG3iVJP5IcWbdzNdgvcaxnG31wH94I"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    if not os.path.exists(model_path):
        with st.spinner('Sedang mengunduh file model AI (500MB+) dari cloud server, mohon tunggu...'):
            gdown.download(url, model_path, quiet=False)
            
    return joblib.load(model_path)

model = load_large_model()
scaler = joblib.load("robust_scaler.pkl")
feature_list = joblib.load("features_list.pkl")


st.markdown('<div class="main-title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instantly evaluate your diabetes risk using trusted artificial intelligence.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2868/2868406.png", width=100)
    st.header("About the App")
    st.write("""
    This application utilizes a **Stacking Machine Learning model** to estimate diabetes risk scores based on your anthropometric metrics, lifestyle factors, and medical history.
    """)
    st.divider()
    st.caption("⚠️ **Important Note:**")
    st.caption("This app is only an initial screening tool. Consult a physician or medical professional for an accurate clinical diagnosis.")


st.subheader("📋 Personal & Health Data")
tab1, tab2, tab3 = st.tabs(["📊 Anthropometrics & Clinical", "🏃‍♂️ Lifestyle", "👤 Demographics"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, help="Weight (kg) / height squared (m²)")
        family_history_diabetes = st.selectbox("Family History of Diabetes", ["No", "Yes"])
    with col2:
        waist_to_hip_ratio = st.slider("Waist-to-Hip Ratio (WHR)", min_value=0.50, max_value=1.50, value=0.85)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        physical_activity_minutes_per_week = st.number_input("Physical Activity (minutes/week)", min_value=0, max_value=1000, value=150)
        diet_score = st.slider("Diet Score", min_value=1.0, max_value=10.0, value=5.0, help="A higher score indicates a healthier dietary pattern")
        sleep_hours_per_day = st.slider("Sleep Duration (hours/day)", min_value=3.0, max_value=12.0, value=7.0)
    with col2:
        alcohol_consumption_per_week = st.number_input("Alcohol Consumption (drinks/week)", min_value=0, max_value=50, value=0)
        screen_time_hours_per_day = st.slider("Screen Time (hours/day)", min_value=0.0, max_value=20.0, value=6.0)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "Other", "White"])
    with col2:
        education = st.selectbox("Education Level", ["No formal", "Highschool", "Graduate", "Postgraduate"])
        income = st.selectbox("Income Level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
        employment = st.selectbox("Employment Status", ["Employed", "Retired", "Student", "Unemployed"])

    st.divider()


    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        predict_btn = st.button("🔍 Calculate Diabetes Risk", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner('Analyzing your metrics with AI...'):
            education_order = {'No formal': 1, 'Highschool': 2, 'Graduate': 3, 'Postgraduate': 4}
            income_order = {'Low': 1, 'Lower-Middle': 2, 'Middle': 3, 'Upper-Middle': 4, 'High': 5}
            smoking_order = {'Never': 1, 'Former': 2, 'Current': 3}

            user_data = {
                "age": age,
                "alcohol_consumption_per_week": alcohol_consumption_per_week,
                "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
                "diet_score": diet_score,
                "sleep_hours_per_day": sleep_hours_per_day,
                "screen_time_hours_per_day": screen_time_hours_per_day,
                "family_history_diabetes": 1 if family_history_diabetes == "Yes" else 0,
                "bmi": bmi,
                "waist_to_hip_ratio": waist_to_hip_ratio,

                "gender_Female": 1 if gender == "Female" else 0,
                "gender_Male": 1 if gender == "Male" else 0,
                "gender_Other": 1 if gender == "Other" else 0,

                "ethnicity_Asian": 1 if ethnicity == "Asian" else 0,
                "ethnicity_Black": 1 if ethnicity == "Black" else 0,
                "ethnicity_Hispanic": 1 if ethnicity == "Hispanic" else 0,
                "ethnicity_Other": 1 if ethnicity == "Other" else 0,
                "ethnicity_White": 1 if ethnicity == "White" else 0,

                "education_level_encoded": education_order[education],
                "income_level_encoded": income_order[income],

                "employment_Employed": 1 if employment == "Employed" else 0,
                "employment_Retired": 1 if employment == "Retired" else 0,
                "employment_Student": 1 if employment == "Student" else 0,
                "employment_Unemployed": 1 if employment == "Unemployed" else 0,

                "smoking_status_encoded": smoking_order[smoking]
            }

            input_df = pd.DataFrame([user_data])
            input_df = input_df.reindex(columns=feature_list, fill_value=0)

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            risk_score = float(prediction * 100) if prediction <= 1.0 else float(prediction)
            risk_score = min(max(risk_score, 0.0), 100.0)

            if risk_score < 30:
                category = "🟢 LOW RISK"
                bg_color = "#D1FAE5"
                text_color = "#065F46"
                advice = "Great job! Maintain your healthy lifestyle by balancing your diet and keeping up with physical activities."
            elif risk_score < 60:
                category = "🟡 MODERATE RISK"
                bg_color = "#FEF3C7"
                text_color = "#92400E"
                advice = "Relatively stable, but stay mindful. Try to increase your physical activity duration and minimize sugary/processed food intake."
            else:
                category = "🔴 HIGH RISK"
                bg_color = "#FEE2E2"
                text_color = "#991B1B"
                advice = "Attention required. It is highly recommended to schedule a routine blood sugar test (Fasting Blood Glucose/HbA1c) at a healthcare facility."


            st.subheader("📊 Risk Assessment Result")
            
            st.progress(risk_score / 100)
            
            st.markdown(f"""
                <div class="result-card" style="background-color: {bg_color}; border: 1px solid {text_color};">
                    <h3 style="color: {text_color}; margin: 0; font-size: 20px;">{category}</h3>
                    <h1 style="color: {text_color}; margin: 10px 0 5px 0; font-size: 48px;">{risk_score:.1f}%</h1>
                    <p style="color: {text_color}; font-size: 14px; font-weight: 500; margin: 0;">Estimated Risk Probability Score</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"💡 **Recommendation:** {advice}")