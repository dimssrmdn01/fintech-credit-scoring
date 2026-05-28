# Fintech Credit Risk & Default Prediction Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![SMOTE](https://img.shields.io/badge/Resampling-SMOTE-success)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)

An end-to-end Machine Learning pipeline designed to assess credit risk and predict loan defaults. This project tackles real-world banking challenges, specifically handling highly imbalanced datasets and providing transparent, explainable AI decisions.

---

## 🚀 Live Demo
Access the live web application here: https://dimas-scoring-engine.streamlit.app/


---

## Project Objectives
In the financial sector, a "black-box" model is unacceptable. This project focuses on two critical aspects of credit scoring:
1.  **Imbalanced Data Handling:** Using SMOTE (Synthetic Minority Over-sampling Technique) to prevent the model from blindly predicting the majority class (non-default).
2.  **Explainable AI (XAI):** Utilizing SHAP (SHapley Additive exPlanations) to transparently explain *why* the model approved or rejected a specific loan application.

## Tech Stack & Methodology
* **Data Engineering & Preprocessing:** `Pandas`, `Scikit-Learn` (StandardScaler)
* **Imbalanced Learning:** `imbalanced-learn` (SMOTE)
* **Predictive Modeling:** `XGBoost` Classifier optimized for `logloss` and recall.
* **Model Interpretability:** `SHAP` TreeExplainer for feature importance and impact analysis.

## Project Structure
```text
fintech-credit-scoring/
│
├── data/
│   ├── raw/                  # Raw, highly imbalanced credit data
│   └── processed/            # Scaled and SMOTE-resampled data (.pkl)
├── docs/
│   └── images/               # SHAP summary plots and class distribution charts
├── models/                   # Serialized XGBoost model
├── Src/
│   ├── 1_data_loader.py         # Data ingestion and imbalance visualization
│   ├── 2_data_preprocessing.py  # Train-test split, scaling, and SMOTE integration
│   └── 3_train_and_explain.py   # Model training, evaluation (ROC-AUC), and SHAP analysis
└── README.md
