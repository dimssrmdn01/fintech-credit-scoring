import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sqlite3
import os

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Fintech Credit Risk Engine", layout="wide")

st.title("Fintech Credit Risk Assessment & Credit Scoring Engine")
st.markdown("### `DECISION STATUS: SECURE` | Automated Risk Evaluation with Explainable AI Framework")
st.markdown("---")

# --- FUNGSI 1: SIMULASI MODEL DATA ---
def predict_credit_risk(features):
    base_score = -1.5 
    risk_from_dti = (features['Debt_to_Income'] / 100) * 2.5
    risk_from_arrears = features['Historical_Arrears'] * 2.0
    risk_from_utilization = (features['Utilization_Rate'] / 100) * 1.5
    benefit_from_income = (features['Annual_Income'] / 100000) * 1.2
    
    total_logit = base_score + risk_from_dti + risk_from_arrears + risk_from_utilization - benefit_from_income
    prob = 1 / (1 + np.exp(-total_logit))
    prediction = 1 if prob > 0.35 else 0 
    return prob, prediction

# --- FUNGSI 2: DATABASE LOGGING ---
def save_log_to_db(name, income, dti, arrears, util, pd_score, decision):
    if not os.path.exists('Database'):
        os.makedirs('Database')
    
    conn = sqlite3.connect('Database/fintech_logs.db')
    cursor = conn.cursor()
    
    # Buat tabel jika belum ada (Jaring pengaman)
    cursor.execute('''CREATE TABLE IF NOT EXISTS applicants (applicant_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, annual_income REAL, dti_ratio REAL, historical_arrears INTEGER, utilization_rate REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS credit_assessments (assessment_id INTEGER PRIMARY KEY AUTOINCREMENT, applicant_id INTEGER, pd_score REAL, decision TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (applicant_id) REFERENCES applicants (applicant_id))''')
    
    cursor.execute('INSERT INTO applicants (name, annual_income, dti_ratio, historical_arrears, utilization_rate) VALUES (?, ?, ?, ?, ?)', (name, income, dti, arrears, util))
    applicant_id = cursor.lastrowid
    cursor.execute('INSERT INTO credit_assessments (applicant_id, pd_score, decision) VALUES (?, ?, ?)', (applicant_id, pd_score, decision))
    
    conn.commit()
    conn.close()

# --- SIDEBAR: INPUT DATA ---
st.sidebar.header("Applicant Metrics Input")
st.sidebar.markdown("Input parameter data calon debitur:")

applicant_name = st.sidebar.text_input("Applicant Full Name", value="John Doe")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
dti = st.sidebar.slider("Debt-to-Income (DTI) Ratio (%)", min_value=0.0, max_value=100.0, value=28.5, step=0.5)
arrears = st.sidebar.number_input("Historical Arrears / Late Payments (Count)", min_value=0, max_value=10, value=0, step=1)
utilization = st.sidebar.slider("Credit Card Utilization Rate (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)

input_features = {
    'Annual_Income': income,
    'Debt_to_Income': dti,
    'Historical_Arrears': arrears,
    'Utilization_Rate': utilization
}

# --- TOMBOL EKSEKUSI ---
tombol_evaluasi = st.sidebar.button("Run Evaluation & Log Data")

if tombol_evaluasi:
    # 1. Jalankan Engine Prediksi
    prob_default, final_decision = predict_credit_risk(input_features)
    
    # 2. Tampilkan Hasil
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Credit Evaluation Summary")
        risk_percentage = prob_default * 100
        
        # Speedometer Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probability of Default (PD)", 'font': {'size': 18}},
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 35], 'color': "lightgreen"},
                    {'range': [35, 70], 'color': "gold"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': risk_percentage}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Kartu Keputusan
        if final_decision == 0:
            st.success(f"✅ APPLICATION APPROVED: Applicant '{applicant_name}' meets the credit safety threshold parameters.")
            status_text = "APPROVED"
        else:
            st.error(f"❌ APPLICATION REJECTED: High risk profile detected. Applicant '{applicant_name}' breaches risk parameters.")
            status_text = "REJECTED"
            
        # Simpan ke Database
        save_log_to_db(applicant_name, income, dti, arrears, utilization, prob_default, status_text)

    with c2:
        st.subheader("Explainable AI (XAI) Feature Importance Matrix")
        st.markdown("Analisis kontribusi fitur (Representasi visual dari SHAP TreeExplainer):")
        
        # Simulasi SHAP Bar Chart
        shap_values = [-(income / 100000), (dti * 0.05), (arrears * 1.5), (utilization * 0.02)]
        features_list = ['Annual Income Impact', 'DTI Ratio Impact', 'Historical Arrears Impact', 'Credit Utilization Impact']
        df_shap = pd.DataFrame({'Features': features_list, 'SHAP Value (Impact)': shap_values}).sort_values(by='SHAP Value (Impact)', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#161b22')
        colors = ['#FF3B30' if x > 0 else '#34C759' for x in df_shap['SHAP Value (Impact)']]
        ax.barh(df_shap['Features'], df_shap['SHAP Value (Impact)'], color=colors, height=0.5)
        ax.axvline(x=0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Risk Contribution Weight (SHAP Scale)", color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=9)
        ax.set_title("Local Explanation: Risk Drivers", color='white', fontsize=11)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

# --- ADMIN PANEL BAWAH ---
st.markdown("---")
with st.expander("🔐 Akses Admin: Lihat Riwayat Database (Log System)"):
    try:
        conn = sqlite3.connect('Database/fintech_logs.db')
        query = '''SELECT a.applicant_id AS "ID", a.name AS "Nama", a.annual_income AS "Pendapatan ($)", a.dti_ratio AS "DTI (%)", c.pd_score AS "Probabilitas Gagal", c.decision AS "Keputusan", c.timestamp AS "Waktu Log" FROM applicants a JOIN credit_assessments c ON a.applicant_id = c.applicant_id ORDER BY c.timestamp DESC'''
        df_logs = pd.read_sql_query(query, conn)
        conn.close()
        st.dataframe(df_logs, use_container_width=True)
    except:
        st.warning("Belum ada data di database. Silakan jalankan evaluasi minimal satu kali.")