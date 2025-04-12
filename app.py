import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import gzip

warnings.filterwarnings('ignore')

# Create directory for storing patient history if it doesn't exist
if not os.path.exists('patient_history'):
    os.makedirs('patient_history')

# Set page config
st.set_page_config(
    page_title="Maternal Risk Classification",
    page_icon="🏥",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        with gzip.open("best_overall_model.pkl.gz", "rb") as file:
            model = pkl.load(file)
        
        with open('scalar.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure both files are in the same directory as the app.")
        st.stop()

# Initialize the model and scaler
model, scaler = load_model()

# Sidebar for patient identification
with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID")
    visit_date = st.date_input("Visit Date", datetime.now())
    
    # Load patient history if available
    def load_patient_history(patient_id):
        try:
            with open(f'patient_history/{patient_id}.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    if patient_id:
        patient_history = load_patient_history(patient_id)
        if patient_history:
            st.success(f"Found {len(patient_history)} previous records")

# Main content
st.title("Maternal Risk Classification System")
st.markdown("""
This application helps healthcare providers assess maternal risk levels based on various health indicators 
and personal factors. Please fill in all the required information below.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Input Data", "Risk Assessment", "Patient History"])

with tab1:
    # Original input fields (same as before)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Vital Signs")
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
        blood_pressure = st.number_input("Blood Pressure (systolic)", 80, 200, 120)
        blood_oxygen = st.number_input("Blood Oxygen (%)", 70, 100, 98)
        fetal_heart_rate = st.number_input("Fetal Heart Rate (bpm)", 100, 200, 140)
        
    with col2:
        st.subheader("Pregnancy Details")
        gestational_age = st.number_input("Gestational Age (weeks)", 0, 45, 28)
        fetal_movement = st.number_input("Fetal Movement Count (24h)", 0, 100, 10)
        previous_complications = st.selectbox("Previous Complications", 
											["None", "Minor", "Major"])
        uterine_contractions = st.selectbox("Uterine Contraction Patterns",
										   ["Normal", "Irregular", "Frequent"])

    with col3:
		st.subheader("Lifestyle Factors")
		physical_activity = st.selectbox("Physical Activity Level",
										["Sedentary", "Light", "Moderate", "Active"])
		dietary_habits = st.selectbox("Dietary Habits",
									 ["Poor", "Fair", "Good", "Excellent"])
		smoking_status = st.selectbox("Smoking Status",
									 ["Never", "Former", "Current"])
		alcohol_consumption = st.selectbox("Alcohol Consumption",
										 ["None", "Occasional", "Regular"])

	
	# Additional factors in expandable sections
	with st.expander("Health Metrics"):
		col4, col5 = st.columns(2)
		with col4:
			hydration = st.selectbox("Hydration Level",
									["Low", "Moderate", "High"])
			sleep_duration = st.number_input("Sleep Duration (hours)", 0, 24, 7)
			stress_level = st.selectbox("Stress Level",
									   ["Low", "Medium", "High"])
		with col5:
			weight_gain = st.number_input("Weight Gain (kg)", 0, 50, 10)
			pre_existing = st.multiselect("Pre-existing Conditions",
										 ["None", "Diabetes", "Hypertension", "Thyroid Disorder"])
			
	with st.expander("Medical History"):
		col6, col7 = st.columns(2)
		with col6:
			prenatal_results = st.selectbox("Prenatal Test Results",
										   ["Normal", "Abnormal", "Not Available"])
			family_history = st.multiselect("Family Medical History",
										   ["None", "Diabetes", "Heart Disease", "Hypertension"])
		with col7:
			current_meds = st.text_area("Current Medications (comma-separated)")
			pain_levels = st.slider("Pain Levels", 0, 10, 2)

	with st.expander("Environmental & Social Factors"):
		col8, col9 = st.columns(2)
		with col8:
			socioeconomic = st.selectbox("Socioeconomic Status",
										["Low", "Middle", "High"])
			healthcare_access = st.selectbox("Access to Healthcare",
										   ["Limited", "Moderate", "Full"])
		with col9:
			prenatal_visits = st.number_input("Frequency of Prenatal Visits (per month)", 0, 10, 1)
			pollutant_exposure = st.selectbox("Exposure to Pollutants",
											 ["Low", "Medium", "High"])

	
	
	
    # [Previous input fields remain the same...]
    # [Include all the input fields from the previous version]

    # Save data button
    if st.button("Save and Predict"):
        if not patient_id:
            st.error("Please enter a Patient ID to save the data")
            st.stop()
            
        # Collect input data (same as before)
        input_data = {
            'date': str(visit_date),
            'heart_rate': heart_rate,
            'blood_pressure': blood_pressure,
            'blood_oxygen': blood_oxygen,
            'fetal_heart_rate': fetal_heart_rate,
            # [Add all other input fields...]
        }
        
        # Process and predict
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Save to patient history
        input_data['prediction'] = prediction
        input_data['prediction_proba'] = prediction_proba.tolist()
        
        patient_history = load_patient_history(patient_id)
        patient_history.append(input_data)
        
        with open(f'patient_history/{patient_id}.json', 'w') as f:
            json.dump(patient_history, f)
        
        st.success("Data saved successfully!")

with tab2:
    if 'prediction' in locals():
        # Create gauge chart for risk level
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = max(prediction_proba) * 100,
            title = {'text': f"Risk Level {prediction}"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge)
        
        # Create bar chart for probability distribution
        fig_proba = px.bar(
            x=[f'Risk Level {i}' for i in range(len(prediction_proba))],
            y=prediction_proba,
            title="Risk Level Probability Distribution"
        )
        st.plotly_chart(fig_proba)
        
        # Display high-risk indicators
        high_risk_indicators = []
        if blood_pressure > 140: high_risk_indicators.append("High blood pressure")
        if fetal_heart_rate < 120 or fetal_heart_rate > 160:
            high_risk_indicators.append("Abnormal fetal heart rate")
        if blood_oxygen < 95: high_risk_indicators.append("Low blood oxygen")
        
        if high_risk_indicators:
            st.warning("⚠️ High Risk Indicators Detected:")
            for indicator in high_risk_indicators:
                st.write(f"- {indicator}")
            st.write("Please consult with a healthcare provider immediately.")

with tab3:
    if patient_id and patient_history:
        # Create timeline of risk levels
        dates = [record['date'] for record in patient_history]
        risk_levels = [record['prediction'] for record in patient_history]
        
        fig_timeline = px.line(
            x=dates,
            y=risk_levels,
            title="Risk Level Timeline",
            labels={'x': 'Date', 'y': 'Risk Level'}
        )
        st.plotly_chart(fig_timeline)
        
        # Show vital signs trends
        vital_signs = ['heart_rate', 'blood_pressure', 'blood_oxygen', 'fetal_heart_rate']
        fig_vitals = go.Figure()
        
        for vital in vital_signs:
            values = [record[vital] for record in patient_history]
            fig_vitals.add_trace(go.Scatter(x=dates, y=values, name=vital.replace('_', ' ').title()))
        
        fig_vitals.update_layout(title="Vital Signs Trends")
        st.plotly_chart(fig_vitals)
        
        # Display previous records in a table
        st.subheader("Previous Records")
        df_history = pd.DataFrame(patient_history)
        st.dataframe(df_history)

# Add footer with disclaimer
st.markdown("---")
st.markdown("""
*Disclaimer: This tool is for informational purposes only and should not replace professional medical advice. 
Always consult with qualified healthcare providers for medical decisions.*
""")