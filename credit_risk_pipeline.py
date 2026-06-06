import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score

# =========================================================
#  DATABASE ENGINE: CREDIT COMPLIANCE AUDIT LAYER
# =========================================================
def init_credit_audit_db():
    conn = sqlite3.connect('credit_audit.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS credit_model_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            imbalance_ratio_awal REAL,
            auc_roc_score REAL,
            recall_score REAL,
            top_driving_feature TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_credit_metrics_to_sql(imbalance_ratio, auc_roc, recall, top_feature):
    conn = sqlite3.connect('credit_audit.db')
    cursor = conn.cursor()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO credit_model_logs (timestamp, imbalance_ratio_awal, auc_roc_score, recall_score, top_driving_feature)
        VALUES (?, ?, ?, ?, ?)
    ''', (waktu_sekarang, float(imbalance_ratio), float(auc_roc), float(recall), str(top_feature)))
    conn.commit()
    conn.close()
    print(f"[REGULATORY COMPLIANCE] Model audit log locked into 'credit_audit.db' 🔒")

# =========================================================
#  SIMULASI DATA RISIKO KREDIT (IMBALANCED DATASET)
# =========================================================
np.random.seed(42)
n_sampel = 1000

# Simulasi ketimpangan data: 90% Bayar Lancar (0), 10% Gagal Bayar/Default (1)
lancar_income = np.random.normal(15, 3, 900)      # Pendapatan lebih tinggi
default_income = np.random.normal(6, 2, 100)       # Pendapatan lebih rendah

lancar_dti = np.random.normal(25, 5, 900)         # Debt-to-Income ratio rendah
default_dti = np.random.normal(55, 12, 100)       # Debt-to-Income ratio tinggi

df_lancar = pd.DataFrame({'Income_Juta': lancar_income, 'DTI_Ratio': lancar_dti, 'Default': 0})
df_default = pd.DataFrame({'Income_Juta': default_income, 'DTI_Ratio': default_dti, 'Default': 1})
df_credit = pd.concat([df_lancar, df_default], ignore_index=True)

# Hitung Rasio Ketimpangan Awal
imbalance_ratio_awal = (len(df_credit[df_credit['Default'] == 1]) / len(df_credit)) * 100

# =========================================================
#  MODEL TRAINING & EXPLAINABILITY EVALUATION
# =========================================================
X = df_credit[['Income_Juta', 'DTI_Ratio']]
y = df_credit['Default']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Eksekusi Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Prediksi Probabilitas Risiko Kredit
y_pred_proba = model_rf.predict_proba(X_val)[:, 1]
y_pred_label = model_rf.predict(X_val)

auc_roc = roc_auc_score(y_val, y_pred_proba)
recall = recall_score(y_val, y_pred_label) # Metrik krusial untuk menangkap risiko gagal bayar

# Ekstraksi Feature Importance (Simulasi Logika Pembuat Keputusan SHAP)
importances = model_rf.feature_importances_
features = X.columns
top_feature = features[np.argmax(importances)]

# =========================================================
#  EXECUTE SQL RISK COMPLIANCE RECORDING
# =========================================================
init_credit_audit_db()
log_credit_metrics_to_sql(imbalance_ratio_awal, auc_roc, recall, top_feature)

print("\n=== 💳 FINTECH CREDIT RISK RISK ASSESSMENT ENGINE ===")
print(f"Rasio Ketimpangan Data Gagal Bayar : {round(imbalance_ratio_awal, 2)}%")
print(f"Metrik Deteksi Gagal Bayar (Recall) : {round(recall * 100, 2)}%")
print(f"Kemampuan Diskriminasi (AUC-ROC)   : {round(auc_roc * 100, 2)}%")
print(f"Fitur Penentu Utama Kelayakan      : {top_feature}")

print("\n" + "="*60)
print("🔍 LIVE QUERY: AUDIT DATA KELAYAKAN KREDIT & BIAS MONITORING FROM SQL")
print("="*60)

conn = sqlite3.connect('credit_audit.db')
df_audit_report = pd.read_sql_query("SELECT * FROM credit_model_logs ORDER BY id DESC LIMIT 5", conn)
conn.close()

print(df_audit_report.to_string(index=False))