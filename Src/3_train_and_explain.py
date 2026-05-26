import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap

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
    # Menggunakan parameter dasar yang umumnya bagus untuk dataset kecil-menengah
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
    # Di perbankan, kita lebih peduli pada Recall kelas 1 (seberapa pintar model menangkap nasabah macet)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # Simpan model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_credit_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel berhasil disimpan di: {model_path}")
    
    # 2. EXPLAINABLE AI (SHAP)
    print("\nMenghasilkan visualisasi SHAP (Explainable AI)...")
    # SHAP Explainer khusus untuk model Tree-based seperti XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Menghitung SHAP values pada sebagian data test agar komputasi tidak terlalu lama
    shap_values = explainer.shap_values(X_test)
    
    # Membuat visualisasi SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, show=False)
    
    # Simpan gambar grafik
    img_path = "docs/images/shap_summary.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Grafik SHAP berhasil disimpan di: {img_path}")
    plt.close()
    
    print("\n🎉 PROSES TRAINING DAN EVALUASI SELESAI!")

if __name__ == "__main__":
    train_and_explain()