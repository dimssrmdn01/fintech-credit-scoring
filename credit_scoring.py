import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

np.random.seed(42)
n_samples = 1000

data = {
    'Pendapatan_Tahunan (Jt)': np.random.normal(120, 30, n_samples),
    'Rasio_Hutang_Pendapatan': np.random.uniform(0.1, 0.8, n_samples),
    'Skor_Kredit': np.random.normal(650, 50, n_samples),
    'Usia': np.random.randint(21, 60, n_samples),
    'Tanggungan': np.random.randint(0, 4, n_samples)
}

df = pd.DataFrame(data)
df['Gagal_Bayar'] = np.where(
    (df['Rasio_Hutang_Pendapatan'] > 0.6) & (df['Skor_Kredit'] < 600), 1, 
    np.where(np.random.rand(n_samples) < 0.15, 1, 0)
)

X = df.drop('Gagal_Bayar', axis=1)
y = df['Gagal_Bayar']

print("=== 💳 FINTECH CREDIT SCORING MODEL ===")
print(f"Distribusi Kelas Awal:\n{y.value_counts()}\n")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Distribusi Kelas Setelah SMOTE:\n{y_resampled.value_counts()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("--- 📊 LAPORAN PERFORMA MODEL (XGBOOST) ---")
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))