import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Auto Insurance Claim Severity & Pure Premium Pricing Engine",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Auto Insurance Claim Severity & Pure Premium Pricing Engine")
st.markdown("**Domain**: `InsurTech / Actuarial ML` | **Tech Stack**: `LightGBM Tweedie, Scikit-Learn, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    policyholder_age_years = st.slider("Policyholder Age Years", float(18.0), float(80.0), float(42.0))
    vehicle_sum_insured_k = st.slider("Vehicle Sum Insured K", float(8.0), float(120.0), float(32.0))
    annual_mileage_km_k = st.slider("Annual Mileage Km K", float(2.0), float(50.0), float(16.0))
    bonus_malus_class = st.slider("Bonus Malus Class", int(0), int(9), int(3))
    territory_risk_score = st.slider("Territory Risk Score", float(0.5), float(2.5), float(1.15))
    prior_claims_3yr = st.slider("Prior Claims 3Yr", int(0), int(4), int(0))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"policyholder_age_years": policyholder_age_years, "vehicle_sum_insured_k": vehicle_sum_insured_k, "annual_mileage_km_k": annual_mileage_km_k, "bonus_malus_class": bonus_malus_class, "territory_risk_score": territory_risk_score, "prior_claims_3yr": prior_claims_3yr}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `pure_claim_cost_usd`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
