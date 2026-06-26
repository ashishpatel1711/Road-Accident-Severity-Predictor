# app.py  — fixed version
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="Road Accident Severity Predictor",
                   layout="wide")
st.title("🚗 Road Accident Severity Predictor")

# ── Load models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler     = joblib.load('models/scaler.pkl')
    feat_names = joblib.load('models/feature_names.pkl')
    xgb        = XGBClassifier()
    xgb.load_model('models/xgboost_accident.json')
    return scaler, xgb, feat_names

scaler, xgb, feat_names = load_models()

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header("Accident conditions")

vehicles     = st.sidebar.slider("Number of vehicles",   1, 10, 2)
casualties   = st.sidebar.slider("Number of casualties", 0, 10, 1)
speed        = st.sidebar.selectbox("Speed limit (mph)", [20, 30, 40, 50, 60, 70])
hour         = st.sidebar.slider("Hour of day", 0, 23, 14)
road_type    = st.sidebar.selectbox("Road type",
                   ["Single carriageway","Dual carriageway","Roundabout","Slip road"])
road_surface = st.sidebar.selectbox("Road surface",
                   ["Dry","Wet","Snow/Ice","Flood"])
weather      = st.sidebar.selectbox("Weather",
                   ["Fine","Raining","Snowing","Fog","High winds"])
light        = st.sidebar.selectbox("Light conditions",
                   ["Daylight","Darkness - lit","Darkness - unlit"])
urban        = st.sidebar.selectbox("Area", ["Urban","Rural"])
weekend      = st.sidebar.checkbox("Weekend", value=False)

# ── Encode inputs ─────────────────────────────────────────────
road_map         = {"Single carriageway":0,"Dual carriageway":1,
                    "Roundabout":2,"Slip road":3}
road_surface_map = {"Dry":0,"Wet":1,"Snow/Ice":2,"Flood":3}
weather_map      = {"Fine":0,"Raining":1,"Snowing":2,"Fog":3,"High winds":4}
light_map        = {"Daylight":0,"Darkness - lit":1,"Darkness - unlit":2}
speed_cat        = 0 if speed<=30 else 1 if speed<=50 else 2 if speed<=70 else 3

# ── Build feature vector aligned to training columns ──────────
row = {col: 0 for col in feat_names}
row['Number_of_Vehicles']       = vehicles
row['Number_of_Casualties']     = casualties
row['Speed_limit']              = speed
row['Road_Type']                = road_map[road_type]
row['Road_Surface_Conditions']  = road_surface_map[road_surface]
row['Weather_Conditions']       = weather_map[weather]
row['Light_Conditions']         = light_map[light]
row['Urban_or_Rural_Area']      = 0 if urban == "Urban" else 1
row['speed_cat']                = speed_cat
row['hour']                     = hour
row['is_weekend']               = int(weekend)

X_input = pd.DataFrame([row])[feat_names]   # correct column order

# ── Predict ───────────────────────────────────────────────────
probs  = xgb.predict_proba(X_input)[0]
pred   = int(np.argmax(probs))
labels = ['Slight', 'Serious', 'Fatal']
colors = ['#4CAF50', '#FF9800', '#F44336']

# ── Display results ───────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("🟢 Slight",  f"{probs[0]*100:.1f}%")
col2.metric("🟠 Serious", f"{probs[1]*100:.1f}%")
col3.metric("🔴 Fatal",   f"{probs[2]*100:.1f}%")

st.markdown(
    f"### Predicted severity: "
    f"<span style='color:{colors[pred]};font-weight:600'>{labels[pred]}</span>",
    unsafe_allow_html=True
)

# Confidence bar
st.progress(int(probs[pred] * 100),
            text=f"Model confidence: {probs[pred]*100:.1f}%")

# ── Heatmap (graceful fallback) ───────────────────────────────
st.divider()
if os.path.exists('outputs/uk_accident_heatmap.html'):
    st.subheader("UK Accident Risk Heatmap")
    st.components.v1.html(
        open('outputs/uk_accident_heatmap.html').read(), height=450)
else:
    st.info("Heatmap not generated yet — run the Phase 9 folium script first.")