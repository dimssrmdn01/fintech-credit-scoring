import sqlite3
import os

def init_db():
    if not os.path.exists('Database'):
        os.makedirs('Database')
        
    # Membuka koneksi (File akan otomatis terbuat jika belum ada)
    conn = sqlite3.connect('Database/fintech_logs.db')
    cursor = conn.cursor()
    
    #Tabel 1: Entitas Pemohon
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS applicants (
            applicant_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            annual_income REAL,
            dti_ratio REAL,
            historical_arrears INTEGER,
            utilization_rate REAL
        )
    ''')
    
    #Tabel 2: Hasil Evaluasi 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS credit_assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id INTEGER,
            pd_score REAL,
            decision TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (applicant_id) REFERENCES applicants (applicant_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Arsitektur Database (fintech_logs.db) berhasil dirakit!")

if __name__ == '__main__':
    init_db()