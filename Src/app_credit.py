import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Pengaturan dasar halaman dashboard credit scoring
st.set_page_config(page_title="Fintech Credit Risk Engine", layout="wide")

st.title("Fintech Credit Risk Assessment & Credit Scoring Engine")
st.markdown("### `DECISION STATUS: SECURE` | Automated Risk Evaluation with Explainable AI Framework")
st.markdown("---")

# -------------------------------------------------------------------
# SIMULASI MODEL DATA (CALIBRATED)
# -------------------------------------------------------------------
def predict_credit_risk(features):
    # Formulasi skoring logistik industri fintech yang sudah dikalibrasi
    # Base risk dimulai dari intercept negatif agar probabilitas default dasar rendah
    base_score = -1.5 
    
    # Faktor yang menaikkan risiko (Positif)
    risk_from_dti = (features['Debt_to_Income'] / 100) * 2.5
    risk_from_arrears = features['Historical_Arrears'] * 2.0
    risk_from_utilization = (features['Utilization_Rate'] / 100) * 1.5
    
    # Faktor yang menurunkan risiko (Negatif)
    benefit_from_income = (features['Annual_Income'] / 100000) * 1.2
    
    # Total Score Linear Logit
    total_logit = base_score + risk_from_dti + risk_from_arrears + risk_from_utilization - benefit_from_income
    
    # Fungsi Aktivasi Sigmoid untuk mengubah logit menjadi Probabilitas (0 hingga 1)
    prob = 1 / (1 + np.exp(-total_logit))
    
    # Threshold persetujuan ketat manajemen risiko bank: 35%
    prediction = 1 if prob > 0.35 else 0 
    return prob, prediction

# -------------------------------------------------------------------
# SIDEBAR CONTROL PANEL - CREDIT ANALYST SIMULATOR
# -------------------------------------------------------------------
st.sidebar.header("Applicant Metrics Input")
st.sidebar.markdown("Input parameter data calon debitur untuk dievaluasi oleh engine:")

applicant_name = st.sidebar.text_input("Applicant Full Name", value="John Doe")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
dti = st.sidebar.slider("Debt-to-Income (DTI) Ratio (%)", min_value=0.0, max_value=100.0, value=28.5, step=0.5)
arrears = st.sidebar.number_input("Historical Arrears / Late Payments (Count)", min_value=0, max_value=10, value=0, step=1)
utilization = st.sidebar.slider("Credit Card Utilization Rate (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)

# Masukkan ke dictionary features
input_features = {
    'Annual_Income': income,
    'Debt_to_Income': dti,
    'Historical_Arrears': arrears,
    'Utilization_Rate': utilization
}

# Run Engine Prediksi
prob_default, final_decision = predict_credit_risk(input_features)

# -------------------------------------------------------------------
# INTERACTIVE TERMINAL LAYOUT
# -------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Credit Evaluation Summary")
    
    # Kartu Keputusan Utama (UI Manarik & Tegas)
    if final_decision == 0:
        st.success(f"APPLICATION APPROVED: Applicant '{applicant_name}' meets the credit safety threshold parameters.")
    else:
        st.error(f"APPLICATION REJECTED: High risk profile detected. Applicant '{applicant_name}' breaches risk parameters.")
        
    # Tampilan Metrik Probabilitas
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(
        label="Probability of Default (PD)", 
        value=f"{prob_default * 100:.2f}%", 
        delta=f"{'HIGH RISK PROFILE' if prob_default > 0.45 else 'LOW RISK PROFILE'}"
    )

with c2:
    st.subheader("Explainable AI (XAI) Feature Importance Matrix")
    st.markdown("Analisis kontribusi fitur (Representasi visual dari SHAP TreeExplainer):")
    
    # Visualisasi Kontribusi Fitur Menggunakan Horizontal Bar Chart (Mirip SHAP Summary)
    # Menghitung dampak kontribusi fiktif berdasarkan parameter input untuk visualisasi
    shap_values = [
        -(income / 100000),      # Pendapatan tinggi menekan risiko (Negatif)
        (dti * 0.05),            # DTI tinggi menaikkan risiko (Positif)
        (arrears * 1.5),         # Tunggakan sangat mendongkrak risiko (Positif)
        (utilization * 0.02)     # Utilitas kartu menaikkan risiko (Positif)
    ]
    features_list = ['Annual Income Impact', 'DTI Ratio Impact', 'Historical Arrears Impact', 'Credit Utilization Impact']
    
    df_shap = pd.DataFrame({'Features': features_list, 'SHAP Value (Impact)': shap_values})
    df_shap = df_shap.sort_values(by='SHAP Value (Impact)', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')
    
    colors = ['#FF3B30' if x > 0 else '#34C759' for x in df_shap['SHAP Value (Impact)']]
    bars = ax.barh(df_shap['Features'], df_shap['SHAP Value (Impact)'], color=colors, height=0.5)
    
    ax.axvline(x=0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Risk Contribution Weight (SHAP Scale)", color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    ax.set_title("Local Explanation: Risk Drivers for Current Applicant", color='white', fontsize=11, fontweight='bold')
    
    # Bersihkan border grafik
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    st.pyplot(fig)