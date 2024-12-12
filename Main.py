import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load CSS for hospital-friendly theme and adjust title positioning
hospital_css = """
<style>
    /* Remove top space and adjust app layout */
    .block-container {
        padding-top: 2px !important;
    }

    html, body {
        background-color: #e0f7fa;  /* Light blue background */
        font-family: Arial, sans-serif;
    }

    h1 {
        color: #006d77;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
    }

    .stButton > button {
        background-color: #008cba;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: #005f73;
    }

    .stNumberInput label {
        color: fffff;
    }

    .stNumberInput input {
        width: 100%;
        max-width: 200px;
    }

    /* Move the select input field slightly upward */
    .stSelectbox {
        margin-top: -10px; /* Adjust the value to move it upward */
    }

    /* Apply background image using a local image URL */
    .stApp {
        background-image: url('https://media-hosting.imagekit.io//1254ea7aea7c44d2/BREAST%20CANCER.png?Expires=1733511859&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=dT8lJoT34CsTdd-tqca87bXt~7vJCPY6Q8yJ37PhJAavEzBW9M-1~L7lw4HygMI-~bsrQk30yUF40Woew6gFAIMjkax3EpKBzoSdi46ptADSd3HuOyTK4h1FVHr4GYFtgPjpG4IuD3gV7~Fn8MP9QLEu~L5hVSGSZnliCPycq81hC1YWe3DediCFWUKNlG~cz42V-bkaYtKKcYqh3PV8b5XsXhcw9ml6KA8WZMKoysWCWVIb2yOWHxy0rL9lxqA~beWZYrVPNOCW-Tacf40pbSyX2a4Pd8mbEtfKFJsvemuNE4HZ8q1lowaakSJJVSU2t9SUudu6UIruKz2mi6WnrQ__'); /* Updated relative path */
        background-size: cover;
        background-position: center;
    }
</style>
"""

# Apply the custom CSS
st.markdown(hospital_css, unsafe_allow_html=True)

# Load the model and scaler
try:
    model_file_path = r'C:\Users\madhu\OneDrive\Documents\MINI-PROJECT(SPEC)\Breast Cancer Prediction Using ML\model.pkl'
    scaler_file_path = r'C:\Users\madhu\OneDrive\Documents\MINI-PROJECT(SPEC)\Breast Cancer Prediction Using ML\scaler.pkl'

    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_file_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Title at the very top
st.title("Breast Cancer Prediction")

# Pre-defined cases for dropdown
malignant_example = [17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                     0.7339, 1.095, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                     25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

benign_example = [12.45, 15.70, 82.57, 477.1, 0.08959, 0.07087, 0.03344, 0.01599, 0.168, 0.05751,
                  0.2524, 0.7681, 1.902, 19.33, 0.005224, 0.01308, 0.01635, 0.005027, 0.01884, 0.003892,
                  14.97, 19.76, 95.50, 686.9, 0.1186, 0.08927, 0.06166, 0.03086, 0.2364, 0.07602]

# Feature names
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Case selection dropdown
case_selection = st.selectbox(
    "Select a case for auto-filling values or choose 'Manual Input':",
    ["Manual Input", "Malignant Example", "Benign Example"]
)

# Initialize inputs
if case_selection == "Malignant Example":
    inputs = malignant_example
elif case_selection == "Benign Example":
    inputs = benign_example
else:
    inputs = [0.0] * 30

# Display inputs in 6 columns
cols = st.columns(6)
user_inputs = []

for i, feature in enumerate(feature_names):
    with cols[i % 6]:  # Cycle through 6 columns
        value = inputs[i] if case_selection != "Manual Input" else 0.0
        user_inputs.append(
            st.number_input(feature, value=value, key=f"input_{i}")
        )

# Submit button
if st.button("Submit"):
    if case_selection == "Manual Input" and any(val == 0.0 for val in user_inputs):
        st.error("Please enter all the values.")
    else:
        input_data = pd.DataFrame([user_inputs], columns=feature_names)

        # Scale the input data
        try:
            input_data_scaled = scaler.transform(input_data)
        except Exception as e:
            st.error(f"Error in scaling input data: {e}")
            st.stop()

        # Make a prediction
        try:
            prediction = model.predict(input_data_scaled)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Display the result
        if prediction[0] == 1:
            st.error("Diagnosis: Malignant")
        else:
            st.success("Diagnosis: Benign")