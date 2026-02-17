import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from Hugging Face
model_path = hf_hub_download(
    repo_id="Anusha3/ab_predictive_maintenance",
    filename="Gradient_Boosting.joblib"
)

# Load model
model = joblib.load(model_path)

# Page config
st.set_page_config(page_title="Predictive Maintenance - Engine Failure")
st.title("Engine Predictive Maintenance System")

st.write("Enter engine parameters below to predict engine condition.")

# ----------------------------
# Input Features (MATCH TRAINING FEATURES EXACTLY)
# ----------------------------

engine_rpm = st.number_input("Engine RPM", value=1500)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", value=5.0)
coolant_pressure = st.number_input("Coolant Pressure", value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", value=80.0)
coolant_temp = st.number_input("Coolant Temperature", value=75.0)

# ----------------------------
# Prepare Input DataFrame
# ----------------------------

input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Engine Condition"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 Engine Failure Likely. Immediate Maintenance Required!")
    else:
        st.success("✅ Engine Operating Normally.")
