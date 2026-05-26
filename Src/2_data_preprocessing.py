import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_and_smote():
    print("Memuat data mentah...")
    input_path = "data/raw/credit_default_raw.csv"
    df = pd.read_csv(input_path)
    
    # Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['Default'])
    y = df['Default']
    
    print(f"Bentuk data awal: Fitur {X.shape}, Target {y.shape}")
    
    # 1. TRAIN-TEST SPLIT (Sangat Krusial!)
    # Kita WAJIB memisahkan data test SEBELUM melakukan SMOTE.
    # Data test harus dibiarkan murni (imbalanced) seperti di dunia nyata agar evaluasi model valid.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Distribusi y_train SEBELUM SMOTE:\n{y_train.value_counts(normalize=True) * 100}")
    
    # 2. SCALING (Standarisasi Fitur Numerik)
    # XGBoost sebenarnya cukup kebal terhadap rentang angka, 
    # tapi scaling sangat membantu algoritma SMOTE bekerja lebih akurat saat mengukur jarak antar data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. MENGAPLIKASIKAN SMOTE HANYA PADA DATA TRAINING
    print("\nMenerapkan SMOTE untuk menyeimbangkan data training...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Distribusi y_train SESUDAH SMOTE:\n{y_train_resampled.value_counts(normalize=True) * 100}")
    
    # 4. MENYIMPAN DATA YANG SUDAH DIPROSES
    # Kita simpan menggunakan format pickle (.pkl) agar siap langsung di-load oleh script training nanti
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    
    processed_data = {
        'X_train': X_train_resampled,
        'y_train': y_train_resampled,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': X.columns.tolist() # Simpan nama kolom asli untuk visualisasi SHAP nanti
    }
    
    output_path = os.path.join(output_dir, "processed_credit_data.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
        
    print(f"\nData yang sudah diproses, di-scale, dan di-SMOTE berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_and_smote()