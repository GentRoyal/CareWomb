import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import warnings
import random
import gzip

import time
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
warnings.filterwarnings('ignore')

# Create directory for storing patient history if it doesn't exist
if not os.path.exists('patient_history'):
    os.makedirs('patient_history')

# Set page config
st.set_page_config(
    page_title="CareWomb",
    page_icon="🏥",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_resources():
    try:
        with gzip.open("best_overall_model.pkl.gz", "rb") as file:
            model = pickle.load(file)
        with open('scalar.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('column_names.pkl', 'rb') as file:
            columns = pickle.load(file)
        
        return model, scaler, columns
        
    except FileNotFoundError:
        st.error("One or more required file not found. Please ensure both files are in the same directory as the app.")
        st.stop()

def preprocess_input(data_dict):
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    df_dummy = pd.get_dummies(df) * 1.0
    
    final_df = pd.DataFrame(0, index = df_dummy.index, columns = expected_columns)
    
    for col in df_dummy.columns:
        if col in expected_columns:
            final_df[col] = df_dummy[col]
    
    
    return final_df

# Function to generate random values
def generate_random_values():
    random_data = {
        'heart_rate': random.randint(60, 120),
        'blood_pressure': random.randint(90, 180),
        'blood_oxygen': random.randint(85, 100),
        'fetal_heart_rate': random.randint(110, 160),
        'gestational_age': random.randint(5, 40),
        'fetal_movement': random.randint(5, 30),
        'previous_complications': random.randint(0, 1),
        'uterine_contractions': random.choice(["Normal", "Irregular", "Frequent"]),
        'physical_activity': random.choice(["Low", "Moderate", "High"]),
        'dietary_habits': random.choice(["Healthy", "Moderate", "Unhealthy"]),
        'smoking_status': random.randint(0, 1),
        'alcohol_consumption': random.randint(0, 1),
        'hydration': random.randint(0, 10),
        'sleep_duration': random.randint(3, 10),
        'stress_level': random.choice(["Low", "Moderate", "High"]),
        'weight_gain': random.randint(0, 20),
        'pre_existing': random.choice(["None", "Diabetes", "Hypertension", "Heart Disease"]),
        'prenatal_results': random.choice(["Normal", "Borderline", "Abnormal"]),
        'family_history': random.choice(["None", "Diabetes", "Hypertension", "Genetic Disorder"]),
        'current_meds': random.choice(["None", "Prenatal Vitamins", "Blood Pressure Meds", "Diabetes Meds"]),
        'pain_levels': random.randint(0, 10),
        'socioeconomic': random.choice(["Low", "Middle", "High"]),
        'healthcare_access': random.choice(["Good", "Moderate", "Poor"]),
        'prenatal_visits': random.randint(0, 12),
        'pollutant_exposure': random.choice(["Low", "Moderate", "High"]),
        'ambient_temperature': random.randint(15, 35),
        'humidity': random.randint(30, 80),
        'ecg_data': round(random.uniform(0.5, 1.5), 2),
        'blood_glucose': random.randint(70, 200),
        'hormone_levels': round(random.uniform(0.0, 10.0), 1)
    }
    return random_data

# Initialize the model and scaler
model, scaler, expected_columns = load_resources()

# Sidebar for patient identification
with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID")
    
    current_date = datetime.now().date()
    current_time = datetime.now().time()

    # Separate inputs for date and time
    visit_date = st.date_input("Date", current_date, disabled=True)
    visit_time = st.time_input("Time", current_time, disabled=True)
    
    visit_date = datetime.combine(visit_date, visit_time)

    
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
st.title("CareWomb App")
st.markdown("""
This application helps healthcare providers assess maternal risk levels based on various health indicators 
and personal factors. Please fill in all the required information below.
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Input Data", "Risk Assessment", "Patient History", "Real-time Monitoring"])

with tab1:
    # Add button to generate random values
    if st.button("Generate Random Values"):
        random_values = generate_random_values()
        # Store random values in session state
        for key, value in random_values.items():
            st.session_state[key] = value
    
    # Original input fields with session state
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Vital Signs")
        heart_rate = st.number_input("Heart Rate (bpm)", 60, 120, 
                                     st.session_state.get('heart_rate', 90))
        blood_pressure = st.number_input("Blood Pressure (systolic)", 90, 180, 
                                         st.session_state.get('blood_pressure', 135))
        blood_oxygen = st.number_input("Blood Oxygen (%)", 85, 100, 
                                       st.session_state.get('blood_oxygen', 92))
        fetal_heart_rate = st.number_input("Fetal Heart Rate (bpm)", 110, 160, 
                                           st.session_state.get('fetal_heart_rate', 135))
		
    with col2:
        st.subheader("Pregnancy Details")
        gestational_age = st.number_input("Gestational Age (weeks)", 5, 40, 
                                         st.session_state.get('gestational_age', 28))
        fetal_movement = st.number_input("Fetal Movement Count (14h)", 5, 30, 
                                        st.session_state.get('fetal_movement', 10))
        previous_complications = st.number_input("Previous Complications (0 = No, 1 = Yes)", 0, 1, 
                                               st.session_state.get('previous_complications', 0))
        
        uterine_contractions = st.selectbox("Uterine Contraction Patterns",
                                           ["Normal", "Irregular", "Frequent"],
                                           index=["Normal", "Irregular", "Frequent"].index(
                                               st.session_state.get('uterine_contractions', "Normal")))

    with col3:
        st.subheader("Lifestyle Factors")
        physical_activity = st.selectbox("Physical Activity Level",
                                        ["Low", "Moderate", "High"],
                                        index=["Low", "Moderate", "High"].index(
                                            st.session_state.get('physical_activity', "Moderate")))
        dietary_habits = st.selectbox("Dietary Habits",
                                     ["Healthy", "Moderate", "Unhealthy"],
                                     index=["Healthy", "Moderate", "Unhealthy"].index(
                                         st.session_state.get('dietary_habits', "Moderate")))
        smoking_status = st.number_input("Smoking Status (0 = No, 1 = Yes)", 0, 1, 
                                        st.session_state.get('smoking_status', 0))
        
        alcohol_consumption = st.number_input("Alcohol Consumption (0 = No, 1 = Yes)", 0, 1, 
                                             st.session_state.get('alcohol_consumption', 0))
	
	# Additional factors in expandable sections
    with st.expander("Health Metrics"):
        col4, col5 = st.columns(2)
        with col4:
            hydration = st.number_input("Hydration Level", 0, 10, 
                                       st.session_state.get('hydration', 5))
									
            sleep_duration = st.number_input("Sleep Duration (hours)", 3, 10, 
                                           st.session_state.get('sleep_duration', 7))
            stress_level = st.selectbox("Stress Level",
                                       ["Low", "Moderate", "High"],
                                       index=["Low", "Moderate", "High"].index(
                                           st.session_state.get('stress_level', "Moderate")))
        with col5:
            weight_gain = st.number_input("Weight Gain (kg)", 0, 20, 
                                         st.session_state.get('weight_gain', 10))
            pre_existing = st.selectbox("Pre-existing Conditions",
                                         ["None", "Diabetes", "Hypertension", "Heart Disease"],
                                         index=["None", "Diabetes", "Hypertension", "Heart Disease"].index(
                                             st.session_state.get('pre_existing', "None")))
			
    with st.expander("Medical History"):
        col6, col7 = st.columns(2)
        with col6:
            prenatal_results = st.selectbox("Prenatal Test Results",
                                           ["Normal", "Borderline", "Abnormal"],
                                           index=["Normal", "Borderline", "Abnormal"].index(
                                               st.session_state.get('prenatal_results', "Normal")))
            family_history = st.selectbox("Family Medical History",
                                           ["None", "Diabetes", "Hypertension", "Genetic Disorder"],
                                           index=["None", "Diabetes", "Hypertension", "Genetic Disorder"].index(
                                               st.session_state.get('family_history', "None")))
        with col7:
            current_meds = st.selectbox("Current Medication", 
                                       ["None", "Prenatal Vitamins", "Blood Pressure Meds", "Diabetes Meds"],
                                       index=["None", "Prenatal Vitamins", "Blood Pressure Meds", "Diabetes Meds"].index(
                                           st.session_state.get('current_meds', "None")))
            pain_levels = st.slider("Pain Levels", 0, 10, 
                                   st.session_state.get('pain_levels', 2))

    with st.expander("Environmental & Social Factors"):
        col8, col9 = st.columns(2)
        with col8:
            socioeconomic = st.selectbox("Socioeconomic Status",
                                        ["Low", "Middle", "High"],
                                        index=["Low", "Middle", "High"].index(
                                            st.session_state.get('socioeconomic', "Middle")))
            healthcare_access = st.selectbox("Access to Healthcare",
                                           ["Good", "Moderate", "Poor"],
                                           index=["Good", "Moderate", "Poor"].index(
                                               st.session_state.get('healthcare_access', "Moderate")))
            prenatal_visits = st.number_input("Frequency of Prenatal Visits (per month)", 0, 12, 
                                             st.session_state.get('prenatal_visits', 1))
            pollutant_exposure = st.selectbox("Exposure to Pollutants",
                                             ["Low", "Moderate", "High"],
                                             index=["Low", "Moderate", "High"].index(
                                                 st.session_state.get('pollutant_exposure', "Low")))
            
        with col9:
            ambient_temperature = st.number_input("Ambient temperature", 15, 35, 
                                                 st.session_state.get('ambient_temperature', 20))
            humidity = st.number_input("Humidity", 30, 80, 
                                      st.session_state.get('humidity', 50))
            ecg_data = st.number_input("Normalized ECG", 0.5, 1.5, 
                                      st.session_state.get('ecg_data', 1.0))
            blood_glucose = st.number_input("Blood Glucose", 70, 200, 
                                           st.session_state.get('blood_glucose', 100))
            hormone_levels = st.slider("Arbitrary Hormone Scale", 0.0, 10.0, 
                                      st.session_state.get('hormone_levels', 5.0))

    # Save data button
    if st.button("Save and Predict"):
        if not patient_id:
            st.error("Please enter a Patient ID to save the data")
            st.stop()
            
        # Collect input data
        input_data = {
            'date': str(visit_date),
            'heart_rate': heart_rate,
            'blood_pressure': blood_pressure,
            'blood_oxygen': blood_oxygen,
            'fetal_heart_rate': fetal_heart_rate,
            'fetal_movement_count': fetal_movement,
            'gestational_age': gestational_age,
            'previous_complications': previous_complications,
            'physical_activity': physical_activity,
            'dietary_habits': dietary_habits,
            'smoking_status': smoking_status,
            'alcohol_consumption': alcohol_consumption,
            'hydration_level': hydration,
            'uterine_contraction_patterns': uterine_contractions,
            'sleep_duration': sleep_duration,
            'stress_level': stress_level,
            'weight_gain': weight_gain,
            'pre_existing_conditions' : pre_existing,
            'prenatal_test_results': prenatal_results,
            'family_medical_history': family_history,
            'current_medications': current_meds,
            'pain_levels': pain_levels,
            'socioeconomic_status': socioeconomic,
            'access_to_healthcare': healthcare_access,
            'freq_prenatal_visits': prenatal_visits,
            'exposure_to_pollutants': pollutant_exposure,
            'ambient_temperature': ambient_temperature,
            'humidity': humidity,
            'ecg_data': ecg_data,
            'blood_glucose': blood_glucose,
            'hormone_levels':hormone_levels
        }
        
        # Process and predict
        processed_data = preprocess_input(input_data)
        
        scaled_data = scaler.transform(processed_data)
        
        prediction = model.predict(scaled_data)[0]
        prediction_proba = model.predict_proba(scaled_data)[0]
        
        
        # Save to patient history
        input_data['prediction'] = prediction
        input_data['prediction_proba'] = prediction_proba.tolist()[int(prediction)]
        
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
    if patient_id and 'patient_history' in locals() and patient_history:
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
        # st.write(patient_history)
        for record in patient_history: 
            if isinstance(record["family_medical_history"], list): 
                record["family_medical_history"] = ", ".join(record["family_medical_history"])
                
            if isinstance(record["pre_existing_conditions"], list): 
                record["pre_existing_conditions"] = ", ".join(record["pre_existing_conditions"])
            
                
        df_history = pd.DataFrame(patient_history)
        st.dataframe(df_history)

with tab4:
    st.subheader("Real-time Patient Monitoring")
    
    # Add real-time monitoring section
    col_monitor1, col_monitor2 = st.columns([1, 3])
    
    with col_monitor1:
        st.markdown("### Patient Vitals")
        st.write("Current status:")
        
        patient_stable = st.checkbox("Patient Stable", value=True)
        monitor_frequency = st.slider("Update Frequency (sec)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        monitoring_duration = st.slider("Monitoring Duration (min)", min_value=1, max_value=60, value=5)
        
        monitor_start = st.button("Start Monitoring")
    
    # Simulate real-time data
    if monitor_start:
        placeholder = st.empty()
        frames = int(monitoring_duration * 60 / monitor_frequency)
        
        # Initialize history data
        history_length = 100  # Keep last 100 data points
        heart_rate_history = [90] * 10
        blood_pressure_history = [120] * 10
        blood_oxygen_history = [98] * 10
        fetal_hr_history = [140] * 10
        
        for i in range(frames):
            # Determine if there should be an anomaly
            if random.random() < 0.05 and not patient_stable:  # 5% chance of anomaly if patient not stable
                anomaly = True
                anomaly_factor = 2.0
            else:
                anomaly = False
                anomaly_factor = 1.0
            
            # Simulate changing vitals
            # Add some realistic patterns: slight upward trend, with some oscillation
            trend_factor = 1 + (i / frames) * 0.1  # Slight upward trend over time
            oscillation = np.sin(i/10) * 3  # Adds a sine wave pattern
            
            # Generate current values with realistic fluctuations
            current_heart_rate = int(90 * trend_factor + oscillation + random.randint(-3, 3) * anomaly_factor)
            current_blood_pressure = int(120 * trend_factor + oscillation + random.randint(-5, 5) * anomaly_factor)
            current_blood_oxygen = min(100, 98 - oscillation/5 + random.randint(-1, 1) * anomaly_factor)
            current_fetal_hr = int(140 + oscillation + random.randint(-4, 4) * anomaly_factor)
            
            # Add to history
            heart_rate_history.append(current_heart_rate)
            blood_pressure_history.append(current_blood_pressure)
            blood_oxygen_history.append(current_blood_oxygen)
            fetal_hr_history.append(current_fetal_hr)
            
            # Keep history at fixed length
            if len(heart_rate_history) > history_length:
                heart_rate_history = heart_rate_history[-history_length:]
                blood_pressure_history = blood_pressure_history[-history_length:]
                blood_oxygen_history = blood_oxygen_history[-history_length:]
                fetal_hr_history = fetal_hr_history[-history_length:]
            
            # Create charts in the placeholder
            with placeholder.container():
                metrics_col1, metrics_col2 = st.columns(2)
                
                # Display current vitals as metrics
                with metrics_col1:
                    st.metric(label="Heart Rate", value=f"{current_heart_rate} bpm", 
                              delta=heart_rate_history[-1]-heart_rate_history[-2])
                    st.metric(label="Blood Pressure", value=f"{current_blood_pressure} mmHg", 
                              delta=blood_pressure_history[-1]-blood_pressure_history[-2])
                
                with metrics_col2:
                    st.metric(label="Blood Oxygen", value=f"{current_blood_oxygen:.1f}%", 
                              delta=round(blood_oxygen_history[-1]-blood_oxygen_history[-2], 1))
                    st.metric(label="Fetal Heart Rate", value=f"{current_fetal_hr} bpm", 
                              delta=fetal_hr_history[-1]-fetal_hr_history[-2])
                
                # Create time axis for charts
                time_axis = list(range(len(heart_rate_history)))
                
                # Create two charts: one for maternal vitals, one for fetal
                chart_maternal, chart_fetal = st.columns(2)
                
                with chart_maternal:
                    st.subheader("Maternal Vitals")
                    maternal_data = pd.DataFrame({
                        'time': time_axis,
                        'Heart Rate': heart_rate_history,
                        'Blood Pressure': blood_pressure_history,
                        'Blood Oxygen': blood_oxygen_history
                    })
                    
                    fig1 = px.line(maternal_data, x='time', y=['Heart Rate', 'Blood Pressure', 'Blood Oxygen'])
                    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with chart_fetal:
                    st.subheader("Fetal Vitals")
                    fetal_data = pd.DataFrame({
                        'time': time_axis,
                        'Fetal Heart Rate': fetal_hr_history
                    })
                    
                    fig2 = px.line(fetal_data, x='time', y='Fetal Heart Rate')
                    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                    fig2.add_hline(y=120, line_dash="dot", line_color="red", annotation_text="Lower Limit")
                    fig2.add_hline(y=160, line_dash="dot", line_color="red", annotation_text="Upper Limit")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Add alert if values go outside normal range
                alert_col1, alert_col2 = st.columns([3, 1])
                with alert_col1:
                    # Check for abnormal values
                    alerts = []
                    if current_heart_rate > 100 or current_heart_rate < 60:
                        alerts.append(f"⚠️ Maternal heart rate is {current_heart_rate} bpm (normal: 60-100)")
                    if current_blood_pressure > 140:
                        alerts.append(f"⚠️ Blood pressure is {current_blood_pressure} mmHg (normal: <140)")
                    if current_blood_oxygen < 95:
                        alerts.append(f"⚠️ Blood oxygen is {current_blood_oxygen:.1f}% (normal: ≥95%)")
                    if current_fetal_hr < 120 or current_fetal_hr > 160:
                        alerts.append(f"⚠️ Fetal heart rate is {current_fetal_hr} bpm (normal: 120-160)")
                    
                    if alerts:
                        for alert in alerts:
                            st.warning(alert)
                    else:
                        st.success("All vitals within normal ranges")
                
                with alert_col2:
                    # Display monitoring info
                    st.info(f"Monitoring {i+1} of {frames} frames")
                    st.progress((i+1)/frames)
            
            # Wait for the specified update frequency
            time.sleep(monitor_frequency)
    
    # Add explanatory text
    st.markdown("""
    ### About Real-time Monitoring
    
    This feature simulates continuous patient monitoring that would connect to actual 
    patient monitoring devices in a clinical setting.
    
    **Features:**
    - Continuous monitoring of maternal heart rate, blood pressure, and blood oxygen
    - Fetal heart rate tracking with reference limits
    - Automatic alerts when vital signs fall outside normal ranges
    - Customizable monitoring frequency and duration
    
    **Clinical Use:**
    - For high-risk pregnancies requiring continuous observation
    - During labor and delivery
    - Post-procedural monitoring
    """)

# Add footer with disclaimer
st.markdown("---")
st.markdown("""
*Disclaimer: This tool is for informational purposes only and should not replace professional medical advice. 
Always consult with qualified healthcare providers for medical decisions.*
""")
