# Predicting Customer Churn for a Telecom Company


# 📦 Telecom Customer Churn Prediction

> 🔍 Predicting customer churn using real-time industry-standard Machine Learning practices.  
> 🚀 Built with XGBoost, SHAP, pandas, Scikit-Learn, and domain-driven feature engineering.

---

## 📌 Project Overview

Telecommunication companies often face significant revenue loss due to customer churn. The goal of this project is to build a **predictive model** that can accurately identify whether a customer is likely to leave the company or not — using their usage patterns and account data.

This **end-to-end machine learning project** covers everything from data ingestion and preprocessing to model training, evaluation, interpretability, and business insights.

---

## 🔧 Technologies Used

| Category           | Tools & Libraries                            |
|-------------------|-----------------------------------------------|
| 📊 Data Handling   | `pandas`, `numpy`, `Power Query (pandas)`     |
| 🧼 ETL             | Custom cleaning, feature engineering, encoding |
| 📉 Modeling        | `XGBoost`, `RandomForest`, `scikit-learn`     |
| 📈 Evaluation      | Accuracy, ROC-AUC, Classification Report      |
| 📊 Visualization   | `matplotlib`, `seaborn`, `SHAP`               |
| 📂 Exporting       | `joblib` for model & scaler storage           |
| 🔁 Real-time Prep  | Colab-ready data pipeline for repeated runs   |
| 💡 Explainability | `SHAP` values and feature importances         |

---

## ✅ Key Features

- **Real-time data aggregation**: Upload raw customer CSV files from any telecom dataset and instantly start analysis.
- **Power Query-like ETL in Python**: Cleaned, transformed, and handled missing and invalid values efficiently.
- **One-hot and label encoding**: Industry-best practices to handle categorical variables.
- **End-to-End ML pipeline**: Training, validation, model tuning, and results generation.
- **Explainable AI**: Visual and numeric SHAP-based feature attribution for transparency.
- **Business-driven insights**: Summary of which features contribute most to churn.
- **Modular and reproducible**: Every step is modularized into Colab cells and can be re-run easily.
- **Deployment-ready model**: Exported `.pkl` files for model and scaler.

---

## 📁 File Structure
📦 Telecom-Customer-Churn-ML/
├── README.md             
├── churn_notebook.ipynb           
├── WA_Fn-UseC_-Telco-Customer-Churn.csv 
│
├── models/                
│   ├── xgb_churn_model.pkl         
│   └── scaler.pkl
│
├── visualizations


---

## 📂 Dataset Source

- 📥 [`WA_Fn-UseC_-Telco-Customer-Churn.csv`](https://www.kaggle.com/blastchar/telco-customer-churn)  
  Provided by IBM Sample Datasets, containing 7043 telecom customer records with 21 features.

---

## 🚀 How to Run

1. **Open in Google Colab** (recommended for GPU/TPU support).
2. Upload your CSV dataset in the second cell.
3. Run all cells sequentially from top to bottom.
4. Trained model and scaler will be saved as `.pkl` files.
5. Modify and extend the notebook to include custom business logic if needed.

---

## 🧠 Model Insights

- Final model used: `XGBoostClassifier`
- Accuracy: ~80-83%
- Key Predictors:
  - MonthlyCharges
  - ContractType
  - Tenure
  - InternetService
  - PaymentMethod
- SHAP & feature importance included for explainability.

---

## 📊 Performance Metrics

| Metric               | Value     |
|----------------------|-----------|
| Accuracy             | ✅ ~83%    |
| ROC AUC Score        | ✅ ~0.85   |
| Precision / Recall   | ✅ Balanced |
| Confusion Matrix     | ✅ Plotted |
| SHAP Summary         | ✅ Yes     |

---

## 📌 Future Improvements

- 📈 Hyperparameter tuning using `GridSearchCV` or `Optuna`
- 📊 Streamlit frontend for real-time predictions
- 🧠 NLP preprocessing for feedback/comments
- 💾 Database integration (MySQL / Firebase)
- 🛠️ AutoML and model selection pipelines

---

## 📚 Learnings & Highlights

- Implemented complete ETL and ML pipeline from scratch.
- Practiced model evaluation and fairness analysis using SHAP.
- Translated business objectives into ML tasks (churn minimization).
- Achieved high-performance predictions without deep learning.

---

## 🙋‍♂️ Author

**Rachakonda Shrutik Sai**  
B.Tech Computer Science, IIIT Dharwad  

---

## ⭐ Contribute

If you like this project and want to contribute:

- 🌟 Star this repo
- 🍴 Fork it and build something awesome
- 📬 Raise issues or suggest features

---

## 🛡️ License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
