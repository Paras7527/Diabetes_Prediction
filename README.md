# 🩺 Diabetes Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview
This repository contains multiple machine learning models to **predict the likelihood of diabetes** based on patient medical data.  

It includes:
- **Logistic Regression Model** for baseline predictions
- **Heterogeneous Ensemble Model** for improved performance
- **Model save/load scripts** for reuse without retraining

The dataset used is `diabetes.csv`, containing medical diagnostic features such as glucose level, BMI, age, etc., along with the diabetes outcome.

---

## 📂 Project Structure
DiabetesPrediction/
│
├── artifacts_sample/ # Stores saved model files
│
├── DiabetesHetroEnsemble/ # Ensemble model implementation
│ ├── artifacts_sample/ # Saved ensemble models
│ ├── diabetes.csv # Dataset copy for ensemble
│ └── EnsembleHetro.py # Heterogeneous ensemble training script
│
├── diabetes.csv # Dataset for Logistic Regression
├── DiabetesModelPreseveTest.py # Script to test saved models
├── LogisticDiabetesPrediction.py # Logistic Regression model training script
└── README.md # Project documentation


---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction


### 2️⃣ Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows


## 📄 Requirements
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0


