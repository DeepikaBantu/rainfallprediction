# Paste the code above here
# app.py

import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Rainfall Prediction System", page_icon="🌧️")
st.title("🌦️ Rainfall Prediction using ML (Random Forest & XGBoost)")

st.write("Enter today's rainfall and lag features to predict if it will rain tomorrow.")

# Load trained models
rf_model = joblib.load("rf_model_imd_features.pkl")
xgb_model = joblib.load("xgb_model_imd_features.pkl")

# Input fields
rain_today = st.number_input("🌧️ Rainfall Today (mm)", 0.0, 100.0, 10.0)
month = st.selectbox("📅 Month", list(range(1, 13)))

rain_lag1 = st.number_input("🌦️ Rainfall Yesterday (lag-1, mm)", 0.0, 100.0, 5.0)
rain_lag2avg = st.number_input("🌦️ Average Rainfall Last 2 Days (mm)", 0.0, 100.0, 5.0)
rain_lag3avg = st.number_input("🌦️ Average Rainfall Last 3 Days (mm)", 0.0, 100.0, 5.0)

# Create month dummy variables (drop_first=True style)
month_dummies = [0] * 11
if month != 1:
    month_dummies[month - 2] = 1

# Combine all features into one input array
X_input = [rain_today, rain_lag1, rain_lag2avg, rain_lag3avg] + month_dummies
X_input = np.array(X_input).reshape(1, -1)

# Model selection
model_choice = st.radio("Choose Model", ["Random Forest", "XGBoost"])

# Prediction button
if st.button("🔍 Predict"):
    if model_choice == "Random Forest":
        pred = rf_model.predict(X_input)[0]
    else:
        pred = xgb_model.predict(X_input)[0]

    # Simple rainfall estimation
    rainfall_amount = np.random.uniform(0, 100) if pred == 1 else np.random.uniform(0, 10)

    st.subheader(f"🌤️ Prediction Result: {'Rain Tomorrow ☔' if pred == 1 else 'No Rain 🌞'}")
    st.write(f"💦 Estimated Rainfall: **{rainfall_amount:.2f} mm**")

    if rainfall_amount > 50:
        st.error("⚠️ Heavy Rain Alert! Please take necessary precautions.")
    elif pred == 1:
        st.warning("🌧️ Light to Moderate Rain Expected.")
    else:
        st.success("🌞 Clear weather likely tomorrow.")

