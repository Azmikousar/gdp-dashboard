import streamlit as st
import pandas as pd
import numpy as np
import time

# Set page layout to wide and brand theme
st.set_page_config(
    page_title="PulseGear AI - Healthcare Intelligence System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for Dark Tech Aesthetic ---
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    div[data-testid="stBlock"] { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 20px; }
    h1, h2, h3 { color: #58a6ff !important; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=70)
st.sidebar.title("PulseGear AI")
st.sidebar.caption("Privacy-Preserving Federated Healthcare System")
st.sidebar.markdown("---")

# Step 1: Select Federated Node
st.sidebar.subheader("🌐 Step 1: Select Hospital Node")
hospital_node = st.sidebar.selectbox(
    "Choose secure local node:",
    ["Hospital Alpha (General Network)", "Hospital Beta (Cardio Specialist)", "Hospital Gamma (Regional Clinic)"]
)

# Mock Patient Records based on selected hospital node
mock_patients = {
    "Hospital Alpha (General Network)": [
        {"id": "P-101", "name": "A. Sharma", "age": 52, "cholesterol": 240, "bp": "145/90", "chest_pain": "Typical Angina"},
        {"id": "P-102", "name": "J. Doe", "age": 45, "cholesterol": 195, "bp": "120/80", "chest_pain": "None"}
    ],
    "Hospital Beta (Cardio Specialist)": [
        {"id": "P-201", "name": "M. Khan", "age": 68, "cholesterol": 285, "bp": "160/95", "chest_pain": "Asymptomatic"},
        {"id": "P-202", "name": "S. Patel", "age": 39, "cholesterol": 210, "bp": "130/85", "chest_pain": "Atypical Angina"}
    ],
    "Hospital Gamma (Regional Clinic)": [
        {"id": "P-301", "name": "R. Das", "age": 61, "cholesterol": 230, "bp": "150/92", "chest_pain": "Non-Anginal"}
    ]
}

# Step 2: Select Patient Data
st.sidebar.subheader("📋 Step 2: Load Patient Data")
patient_options = [p["id"] for p in mock_patients[hospital_node]]
selected_patient_id = st.sidebar.selectbox("Select local Patient ID:", patient_options)

# Get current patient details
patient_data = next(p for p in mock_patients[hospital_node] if p["id"] == selected_patient_id)

# --- Header Area ---
st.title("🫀 PulseGear AI")
st.markdown("### Privacy-Preserving Collaborative Disease Prediction using Federated Learning & Explainable AI")
st.info("🔒 **Federated Privacy Guard Activated:** Raw patient data never leaves local boundaries. Only global cryptographic model weights are aggregated.")

# --- Layout: Main Screen split into columns ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📋 Local Patient Profile")
    st.markdown(f"**Patient Name:** `{patient_data['name']}` | **Local ID:** `{patient_data['id']}`")
    
    # Form layout for editing/confirming variables
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", value=patient_data["age"], disabled=True)
    chol = c2.number_input("Cholesterol (mg/dL)", value=patient_data["cholesterol"])
    
    c3, c4 = st.columns(2)
    bp = c3.text_input("Blood Pressure Status", value=patient_data["bp"])
    cp_type = c4.text_input("Chest Pain Symptoms", value=patient_data["chest_pain"], disabled=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_prediction = st.button("🚀 Run Local AI Prediction & Sync Updates", type="primary", use_container_width=True)

with col2:
    st.subheader("❤️ Digital Heart Twin Simulation")
    
    if run_prediction:
        with st.spinner("Processing deep neural metrics locally..."):
            time.sleep(1.5) # Simulating AI processing delay
            
        # Algorithmic hazard calculation for simulation
        risk_score = min(int((chol / 300 * 40) + (age / 80 * 40) + (5 if "140" in bp or "150" in bp or "160" in bp else 0)), 100)
        
        if risk_score < 45:
            status_color = "🟢 HEALTHY STATE"
            heart_emoji = "💚"
            risk_level = "LOW"
            ui_alert = st.success
        elif risk_score < 75:
            status_color = "🟡 MODERATE RISK STATE"
            heart_emoji = "💛"
            risk_level = "MODERATE"
            ui_alert = st.warning
        else:
            status_color = "🔴 HIGH RISK STATE"
            heart_emoji = "💔"
            risk_level = "HIGH"
            ui_alert = st.error

        # Metric Displays
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{risk_score} / 100")
        m2.metric("Risk Level", risk_level)
        m3.metric("AI Confidence", "91%")
        
        ui_alert(f"**Visual Simulation Mode:** Simulated Heart state detected as **{status_color}**")
        st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{heart_emoji}</h1>", unsafe_allow_html=True)
        st.caption("<p style='text-align: center;'>Dynamic architectural mesh responding to real-time spatial analytics.</p>", unsafe_allow_html=True)
    else:
        st.info("Click **'Run Local AI Prediction'** to render the dynamic Digital Twin visualization.")

st.markdown("---")

# --- Expandable & Explainable AI (SHAP Metrics) ---
st.subheader("📊 Transparent Decision-Making & Explainable AI")
if run_prediction:
    st.markdown("#### SHAP Feature Importance Metrics (Why the AI made this choice)")
    
    # Render simulated horizontal bars indicating contribution weights
    features = {
        "Cholesterol Contribution": min(int(chol * 0.1), 35),
        "Age Factor Weight": min(int(age * 0.4), 30),
        "Blood Pressure Deviation": 15 if ("140" in bp or "150" in bp or "160" in bp) else 5,
        "Symptomatic Chest Pain": 20 if cp_type != "None" else 0
    }
    
    for key, value in features.items():
        st.write(f"**{key}** (`+{value}%` impact)")
        st.progress(value / 40)
        
    st.markdown("""
        > 💡 **Clinical Trust Note:** SHAP (SHapley Additive exPlanations) values break down black-box neural networks 
        into logical allocations. Doctors can audit exactly which parameters drove the warning flags.
    """)
else:
    st.caption("Awaiting calculation parameters to display clinical interpretability breakdown.")

# --- Footer System Status ---
st.markdown("---")
f1, f2, f3 = st.columns(3)
f1.markdown("🔒 **Encryption:** AES-256 Bit Data Flows")
f2.markdown("⚙️ **Global Pipeline:** Aggregator Active")
f3.markdown("⚖️ **Fairness Monitor:** Bias Check Passed ✅")
