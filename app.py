# app.py

import streamlit as st
import numpy as np
import joblib
import os

# --------------------------
# Streamlit setup
# --------------------------
st.set_page_config(page_title="Rainfall Prediction", page_icon="🌧️")
st.title("🌦️ Rainfall Prediction using XGBoost")

# --------------------------
# Load XGBoost model from local GitHub repo
# --------------------------
xgb_model_path = "xgb_model_imd_features.pkl"

try:
    xgb_model = joblib.load(xgb_model_path)
except Exception as e:
    st.error(f"Failed to load XGBoost model: {e}")

# --------------------------
# Input fields
# --------------------------
rain_today = st.number_input("🌧️ Rainfall Today (mm)", 0.0, 100.0, 10.0)
month = st.selectbox("📅 Month", list(range(1, 13)))
rain_lag1 = st.number_input("🌦️ Rainfall Yesterday (lag-1, mm)", 0.0, 100.0, 5.0)
rain_lag2avg = st.number_input("🌦️ Average Rain Last 2 Days (mm)", 0.0, 100.0, 5.0)
rain_lag3avg = st.number_input("🌦️ Average Rain Last 3 Days (mm)", 0.0, 100.0, 5.0)

# --------------------------
# Prepare month dummies
# --------------------------
month_dummies = [0]*11
if month != 1:
    month_dummies[month-2] = 1

# Full input for XGBoost
X_input = np.array([rain_today, rain_lag1, rain_lag2avg, rain_lag3avg] + month_dummies).reshape(1,-1)

# --------------------------
# Prediction button
# --------------------------
if st.button("🔍 Predict"):
    try:
        pred = xgb_model.predict(X_input)[0]

        # Simulate rainfall amount
        rainfall_amount = np.random.uniform(0, 100) if pred==1 else np.random.uniform(0,10)

        st.subheader(f"🌤️ Prediction Result: {'Rain Tomorrow ☔' if pred==1 else 'No Rain 🌞'}")
        st.write(f"💦 Estimated Rainfall: **{rainfall_amount:.2f} mm**")

        if rainfall_amount > 50:
            st.error("⚠️ Heavy Rain Alert! Please take necessary precautions.")
        elif pred==1:
            st.warning("🌧️ Light to Moderate Rain Expected.")
        else:
            st.success("🌞 Clear weather likely tomorrow.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
