import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Anusha3/ab_predictive_maintenance", filename="Gradient_Boosting.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Predictive Maintence Prediction
st.set_page_config(page_title="Predictive Maintenance - Engine Failure")
st.title("Aircraft Engine Predictive Maintenance")

st.write("Enter sensor readings below to predict engine failure probability.")

# ----------------------------
# Input Features (Based on Engine Dataset)
# ----------------------------

operational_setting_1 = st.number_input("Operational Setting 1", value=0.0)
operational_setting_2 = st.number_input("Operational Setting 2", value=0.0)
operational_setting_3 = st.number_input("Operational Setting 3", value=0.0)

sensor_1 = st.number_input("Sensor Measurement 1", value=0.0)
sensor_2 = st.number_input("Sensor Measurement 2", value=0.0)
sensor_3 = st.number_input("Sensor Measurement 3", value=0.0)
sensor_4 = st.number_input("Sensor Measurement 4", value=0.0)
sensor_5 = st.number_input("Sensor Measurement 5", value=0.0)
sensor_6 = st.number_input("Sensor Measurement 6", value=0.0)
sensor_7 = st.number_input("Sensor Measurement 7", value=0.0)
sensor_8 = st.number_input("Sensor Measurement 8", value=0.0)
sensor_9 = st.number_input("Sensor Measurement 9", value=0.0)
sensor_10 = st.number_input("Sensor Measurement 10", value=0.0)

# ----------------------------
# Prepare Input DataFrame
# ----------------------------

input_data = pd.DataFrame([{
    "operational_setting_1": operational_setting_1,
    "operational_setting_2": operational_setting_2,
    "operational_setting_3": operational_setting_3,
    "sensor_1": sensor_1,
    "sensor_2": sensor_2,
    "sensor_3": sensor_3,
    "sensor_4": sensor_4,
    "sensor_5": sensor_5,
    "sensor_6": sensor_6,
    "sensor_7": sensor_7,
    "sensor_8": sensor_8,
    "sensor_9": sensor_9,
    "sensor_10": sensor_10
}])

# ----------------------------
# Prediction Section
# ----------------------------

if st.button("Predict Engine Failure"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 High Risk: Engine Failure Likely. Immediate Maintenance Recommended.")
    else:
        st.success("✅ Engine Operating Normally. No Immediate Maintenance Required.")
