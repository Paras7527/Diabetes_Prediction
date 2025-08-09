# 🩺 Diabetes Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview
This repository contains multiple machine learning models to **predict the likelihood of diabetes** based on patient medical data.  

It includes:
- **Logistic Regression Model** for baseline predictions.
- **Heterogeneous Ensemble Model** for improved performance.
- **Model save/load scripts** for reuse without retraining.

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
│ ├── EnsembleHetro.py # Heterogeneous ensemble training script
│
├── diabetes.csv # Dataset for Logistic Regression
├── DiabetesModelPreseveTest.py # Script to test saved models
├── LogisticDiabetesPrediction.py # Logistic Regression model training script

yaml
Copy
Edit

---

## ⚙️ Installation
1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
2️⃣ Create and activate a virtual environment (recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
📄 Requirements
shell
Copy
Edit
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
ipykernel>=6.23.0
jupyter>=1.0.0
scipy>=1.10.0
🚀 Usage
Train Logistic Regression Model
bash
Copy
Edit
python LogisticDiabetesPrediction.py
Loads diabetes.csv

Trains a Logistic Regression model

Saves the trained model in artifacts_sample/DiabetesPred.joblib

Train Ensemble Model
bash
Copy
Edit
cd DiabetesHetroEnsemble
python EnsembleHetro.py
Loads diabetes.csv from DiabetesHetroEnsemble/

Builds a heterogeneous ensemble of classifiers

Saves model in artifacts_sample/

Test Saved Model
bash
Copy
Edit
python DiabetesModelPreseveTest.py
Loads saved .joblib model

Runs prediction on sample data

📊 Dataset Information
Features:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

Target:

Outcome → 0 = No Diabetes, 1 = Diabetes

📈 Example Output
yaml
Copy
Edit
Data Loaded Successfully!!!
The Initial data in the dataset :
   Pregnancies  Glucose  BloodPressure ...
0            6      148            72 ...
1            1       85            66 ...
...
Accuracy is :  0.785
Model saved successfully at: artifacts_sample/DiabetesPred.joblib
