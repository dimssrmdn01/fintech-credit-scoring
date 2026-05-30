import sqlite3
from datetime import datetime

DB_NAME = "portfolio_logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Tabel 1: Sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME
        )
    ''')
    
    # Tabel 2: Credit Inputs (Disesuaikan dengan fitur kodemu)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS credit_inputs (
            input_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            applicant_name TEXT,
            annual_income REAL,
            dti_ratio REAL,
            historical_arrears INTEGER,
            utilization_rate REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
    ''')
    
    # Tabel 3: Predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_id INTEGER,
            pd_score REAL,
            final_decision INTEGER,
            FOREIGN KEY(input_id) REFERENCES credit_inputs(input_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def log_prediction(session_id, inputs, pd_score, final_decision):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute('INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)', 
                       (session_id, datetime.now()))
        
        cursor.execute('''
            INSERT INTO credit_inputs (session_id, applicant_name, annual_income, dti_ratio, historical_arrears, utilization_rate) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, inputs['name'], inputs['income'], inputs['dti'], inputs['arrears'], inputs['utilization']))
        input_id = cursor.lastrowid 
        
        cursor.execute('''
            INSERT INTO predictions (input_id, pd_score, final_decision) 
            VALUES (?, ?, ?)
        ''', (input_id, pd_score, final_decision))
        
        conn.commit()
    except Exception as e:
        print(f"Error logging to DB: {e}")
    finally:
        conn.close()