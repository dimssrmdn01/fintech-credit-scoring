import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def fetch_and_analyze_data():
    print("Mendownload dataset Credit Risk...")
    url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Credit%20Default.csv"
    
    try:
        df = pd.read_csv(url)
        print("Dataset berhasil dimuat!")
        
        #Simpan raw data
        raw_path = "data/raw/credit_default_raw.csv"
        df.to_csv(raw_path, index=False)
        print(f"Data mentah disimpan di: {raw_path}")
        
        print("\n--- INFO DATASET ---")
        print(f"Jumlah Baris: {df.shape[0]}")
        print(f"Jumlah Kolom: {df.shape[1]}")
        
        #Mengecek rasio ketimpangan pada target variable 
        target_col = 'Default'
        imbalance = df[target_col].value_counts(normalize=True) * 100
        
        print("\n--- KETIMPANGAN KELAS (CLASS IMBALANCE) ---")
        print(f"Nasabah Lancar (0): {imbalance[0]:.2f}%")
        print(f"Nasabah Gagal Bayar (1): {imbalance[1]:.2f}%")
        
        #Visualisasi Ketimpangan
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=target_col, palette='Set2')
        plt.title('Ketimpangan Data Nasabah: Lancar vs Gagal Bayar')
        plt.xlabel('Status (0 = Lancar, 1 = Gagal Bayar)')
        plt.ylabel('Jumlah Nasabah')
        
        #Simpan grafik
        img_path = "docs/images/class_imbalance.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"\nGrafik ketimpangan data berhasil disimpan di: {img_path}")
        plt.close()
        
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat data: {e}")

if __name__ == "__main__":
    fetch_and_analyze_data()
