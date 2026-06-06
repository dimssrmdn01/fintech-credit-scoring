# 💳 Fintech Credit Risk & Default Prediction Engine (Pro Version)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![SMOTE](https://img.shields.io/badge/Resampling-SMOTE-success)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)
![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey)

An end-to-end Machine Learning pipeline and risk compliance tracking system designed to assess credit risk and predict loan defaults. This project handles highly imbalanced default rates using synthetic resampling, provides transparent decision-making paths using Shapley values, and locks historical validation performance into a persistent SQL data layer for financial audit readiness.

---

## 🚀 Live Demo
Access the live web application here: https://dimas-scoring-engine.streamlit.app/

---

## 📁 Project Structure

```text
fintech-credit-scoring/
│
├── data/
│   ├── raw/                  # Raw, highly imbalanced credit data
│   └── processed/            # Scaled and SMOTE-resampled training partitions (.pkl)
├── docs/
│   └── images/               # SHAP summary plots and class distribution charts
├── models/
│   ├── credit_audit.db       # Relational SQL audit logging database for compliance
│   └── xgboost_model_tuned.pkl # Serialized production-ready XGBoost artifacts
├── Src/
│   ├── 1_data_loader.py         # Data ingestion and imbalance visualization
│   ├── 2_data_preprocessing.py  # Train-test split, scaling, and SMOTE integration
│   └── 3_train_and_explain.py   # Model training, SHAP value extraction, and SQL logging
├── config.yaml               # Global parameters and hyperparameters
└── README.md                 # Project documentation
```

---

## 🛠️ Core Methodologies & Architecture

1. **Imbalanced Learning Data Layer:** Implements `SMOTE` (Synthetic Minority Over-sampling Technique) to synthetically balance historical loan default occurrences, preventing model classification bias toward high-volume non-default profiles.
2. **Gradient Boosting Resolution:** Trains an `XGBoost Classifier` fine-tuned to map non-linear financial interactions and optimized against strict recall boundaries to minimize credit default leakages.
3. **Explainable AI (XAI Audit):** Deploys `SHAP (SHapley Additive exPlanations)` TreeExplainer mechanics to decompose specific feature risk-contributions, creating fully transparent justification metrics for legal auditing.
4. **Persistent Compliance Archiving:** Automatically intercept valuation metrics and commits logs into an `SQLite3` table database (`credit_audit.db`) upon pipeline execution.

---

## 🗄️ SQL Model Governance Schema

Every operational iteration records governance vectors directly into the persistent storage engine:

| Database Parameter | Storage Class | Risk Management Focus |
| :--- | :--- | :--- |
| **timestamp** | TEXT | Verifies model operational timeline synchronization for corporate compliance audits. |
| **model_name** | TEXT | Documents target training configuration (e.g., XGBoost + SMOTE framework). |
| **auc_roc** | REAL | Measures macro-discriminatory baseline accuracy under volatile stress conditions. |
| **recall** | REAL | Primary metric evaluating the pipeline's exact precision in capturing risky defaulting accounts. |
| **top_shap_feature** | TEXT | Logs the most impactful risk vector driving credit decisions to map model drift. |

---

## ⚙️ How to Run (Local Setup)

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/dimssrmdn01/fintech-credit-scoring.git](https://github.com/dimssrmdn01/fintech-credit-scoring.git)
   cd fintech-credit-scoring
   ```

2. **Install Core Dependencies:**
   ```bash
   pip install xgboost shap scikit-learn pandas numpy imbalanced-learn
   ```

3. **Execute the Modular Pipeline:**
   ```bash
   python Src/1_data_loader.py
   python Src/2_data_preprocessing.py
   python Src/3_train_and_explain.py
   ```
