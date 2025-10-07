# app.py

import streamlit as st
import numpy as np
import joblib
import gdown
import os

# --------------------------
# Download models if not present
# --------------------------
rf_model_path = "rf_model_imd_features.pkl"
xgb_model_path = "xgb_model_imd_features.pkl"

if not os.path.exists(rf_model_path):
    gdown.download(
        "https://drive.google.com/uc?id=1mkggx9pV0_cz96Zy--KBacyXFjfdPWTa",
        rf_model_path,
        quiet=False
    )

if not os.path.exists(xgb_model_path):
    gdown.download(
        "https://drive.google.com/uc?id=1ELOOh_li1HkECS24J78pcmRhIABfi_To",
        xgb_model_path,
        quiet=False
    )

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="Rainfall Prediction System", page_icon="🌧️")
st.title("🌦️ Rainfall Prediction using ML (Random Forest & XGBoost)")

st.write("Enter today's rainfall and lag features to predict if it will rain tomorrow.")

# --------------------------
# Load trained models
# --------------------------
rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)

# --------------------------
# User Input
# --------------------------
rain_today = st.number_input("🌧️ Rainfall Today (mm)", 0.0, 100.0, 10.0)
month = st.selectbox("📅 Month", list(range(1, 13)))
rain_lag1 = st.number_input("🌦️ Rainfall Yesterday (lag-1, mm)", 0.0, 100.0, 5.0)
rain_lag2avg = st.number_input("🌦️ Average Rainfall Last 2 Days (mm)", 0.0, 100.0, 5.0)
rain_lag3avg = st.number_input("🌦️ Average Rainfall Last 3 Days (mm)", 0.0, 100.0, 5.0)

# --------------------------
# Prepare features
# --------------------------
# Month dummy variables (drop_first=True style)
month_dummies = [0] * 11
if month != 1:
    month_dummies[month - 2] = 1

# Full input for XGBoost (15 features)
X_input_full = [rain_today, rain_lag1, rain_lag2avg, rain_lag3avg] + month_dummies
X_input_full = np.array(X_input_full).reshape(1, -1)

# Input for Random Forest (use only features RF was trained on)
# Check RF feature count
rf_features = rf_model.n_features_in_
if rf_features == 1:
    X_input_rf = np.array([rain_today]).reshape(1, -1)
else:
    # If RF was trained with more features, take full input
    X_input_rf = X_input_full

# --------------------------
# Model selection
# --------------------------
model_choice = st.radio("Choose Model", ["Random Forest", "XGBoost"])

# --------------------------
# Prediction
# --------------------------
if st.button("🔍 Predict"):
    if model_choice == "Random Forest":
        pred = rf_model.predict(X_input_rf)[0]
    else:
        pred = xgb_model.predict(X_input_full)[0]

    # Simple rainfall estimation (dummy)
    rainfall_amount = np.random.uniform(0, 100) if pred == 1 else np.random.uniform(0, 10)

    # Display results
    st.subheader(f"🌤️ Prediction Result: {'Rain Tomorrow ☔' if pred == 1 else 'No Rain 🌞'}")
    st.write(f"💦 Estimated Rainfall: **{rainfall_amount:.2f} mm**")

    if rainfall_amount > 50:
        st.error("⚠️ Heavy Rain Alert! Please take necessary precautions.")
    elif pred == 1:
        st.warning("🌧️ Light to Moderate Rain Expected.")
    else:
        st.success("🌞 Clear weather likely tomorrow.")
import streamlit as st
st.title("Test Streamlit")
st.write("Hello, Streamlit is working!")
streamlit run test_app.py
        
streamlit run app.py
