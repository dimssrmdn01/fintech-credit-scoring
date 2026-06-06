import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, confusion_matrix
import shap
import sqlite3
from datetime import datetime

# =========================================================
#  DATABASE ENGINE: CREDIT COMPLIANCE AUDIT LAYER
# =========================================================
def init_credit_audit_db():
    os.makedirs("models", exist_ok=True)
    conn = sqlite3.connect('models/credit_audit.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS credit_compliance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            auc_roc REAL,
            recall REAL,
            top_shap_feature TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_compliance_metrics(model_name, auc_roc, recall, top_feature):
    conn = sqlite3.connect('models/credit_audit.db')
    cursor = conn.cursor()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO credit_compliance_logs (timestamp, model_name, auc_roc, recall, top_shap_feature)
        VALUES (?, ?, ?, ?, ?)
    ''', (waktu_sekarang, model_name, float(auc_roc), float(recall), str(top_feature)))
    conn.commit()
    conn.close()
    print(f"[REGULATORY AUDIT] Metrics & SHAP explainability successfully locked into SQL database! 🔒")

# =========================================================
# MAIN TRAINING AND EXPLAINABILITY PIPELINE
# =========================================================
def train_and_explain():
    print("Memuat data yang sudah diproses...")
    input_path = "data/processed/processed_credit_data.pkl"
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print("Melatih model XGBoost dengan data hasil SMOTE...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    print("\n--- EVALUASI MODEL PADA DATA TEST (DUNIA NYATA) ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report & ROC-AUC
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # Tambahan kalkulasi recall untuk kebutuhan pelaporan audit risiko kredit
    rec_score = recall_score(y_test, y_pred, zero_division=0)
    
    # Simpan model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_credit_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel berhasil disimpan di: {model_path}")
    
    # 2. EXPLAINABLE AI (SHAP)
    print("\nMenghasilkan visualisasi SHAP (Explainable AI)...")
    explainer = shap.TreeExplainer(model)
    
    # Menghitung SHAP values pada sebagian data test 
    shap_values = explainer.shap_values(X_test)
    
    # Ekstraksi otomatis fitur paling dominan berdasarkan nilai absolut rata-rata SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_idx = np.argmax(mean_abs_shap)
    top_shap_feature = feature_names[top_feature_idx]
    
    # Membuat visualisasi SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, show=False)
    
    # Simpan gambar grafik
    img_path = "docs/images/shap_summary.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Grafik SHAP berhasil disimpan di: {img_path}")
    plt.close()
    
    # =========================================================
    #  SUNTIKAN BARU: COMMIT DAN LIVE QUERY LAPORAN AUDIT
    # =========================================================
    init_credit_audit_db()
    log_compliance_metrics('XGBoost + SMOTE Pipeline', auc_score, rec_score, top_shap_feature)
    
    print("\n" + "="*60)
    print("🔍 LIVE QUERY: DATA REKOR KEPATUHAN RISIKO FINTECH (TABEL SQL)")
    print("="*60)
    conn = sqlite3.connect('models/credit_audit.db')
    df_sql_log = pd.read_sql_query("SELECT * FROM credit_compliance_logs ORDER BY id DESC LIMIT 5", conn)
    conn.close()
    print(df_sql_log.to_string(index=False))
    
    print("\n PROSES TRAINING DAN EVALUASI SELESAI!")

if __name__ == "__main__":
    train_and_explain()