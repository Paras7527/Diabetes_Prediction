#############################################
# Required Python Packages
#############################################
from pathlib import Path    
import joblib     

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

#############################################
# File Paths
#############################################
INPUT_PATH = "diabetes.data"
OUTPUT_PATH = "diabetes.csv"
MODEL_PATH = "DiabetesPred.joblib"

#############################################
# Headers
#############################################

HEADERS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

#############################################
# Function name :    read_data
# Description :      Read the data into pandas dataframe
# Input :            path of CSV file
# Output :           Gives the data
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def read_data(path):
    """Read the data into pandas dataframe"""
    data = pd.read_csv(path, header=None)
    return data

#############################################
# Function name :    get_headers
# Description :      dataset headers
# Input :            dataset
# Output :           Returns the header
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def get_headers(dataset):
    """Return dataset headers"""
    return dataset.columns.values

#############################################
# Function name :    add_headers
# Description :      Add the headers to the dataset
# Input :            dataset
# Output :           Updated dataset
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def add_headers(dataset, headers):
    """Add headers to dataset"""
    dataset.columns = headers
    return dataset

#############################################
# Function name :    data_file_to_csv
# Description :      Nothing
# Input :            Nothing
# Output :           Write the data to CSV
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def data_file_to_csv():
    """Convert raw .data file to CSV with headers"""
    dataset = read_data(INPUT_PATH)
    dataset = add_headers(dataset, HEADERS)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("File saved ...!")


#############################################
# Function name :    split_dataset
# Description :      Split the dataset with train_percentage
# Input :            Dataset with related information
# Output :           Dataset after splitting
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def split_dataset(dataset, train_percentage, feature_headers, target_header, random_state=42):
    """Split dataset into train/test"""
    train_x, test_x, train_y, test_y = train_test_split(
        dataset[feature_headers], dataset[target_header],
        train_size=train_percentage, random_state=random_state,
        stratify=dataset[target_header]
    )

    return train_x, test_x, train_y, test_y

#############################################
# Function name :    dataset_statistics
# Description :      Display the statistics
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def dataset_statistics(dataset):
    """Print basic stats"""
    print(dataset.describe(include='all'))

#############################################
# Function name :    build_pipeline
# Description :      Build a Pipeline:
#                    SimpleImputer: replace missing with median
#                    RandomForestClassifier: robust baseline
# Author :           Piyush Manohar Khairnar
# Date :             09/08/2025
#############################################

def build_pipeline():

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("lr", LogisticRegression(
            max_iter=100
        ))
    ])

    return pipe

#############################################
# Function name :    train_pipeline
# Description :      Train a Pipeline:
# Author :           Piyush Manohar Khairnar
# Date :             09/08/2025
#############################################

def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


#############################################
# Function name :    save_model
# Description :      Save the model
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


#############################################
# Function name :    load_model
# Description :      Load the trained model
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model



#############################################
# Function name :    plot_confusion_matrix_matshow
# Description :      Display Confusion Matrix
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################

def plot_confusion_matrix_matshow(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

#############################################
# Function name :    plot_feature_importances
# Description :      Display the feature importance
# Author :           Paras Shivprasad kulkarni
# Date :             11/08/2025
#############################################

def plot_feature_importances(model, feature_names, title="Feature Importances (Random Forest)"):
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Feature importances not available for this model.")
        return

    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

#############################################
# Function name :    main
# Description :      Main function from where execution starts
# Author :           Paras Shivprasad Kulkarni
# Date :             11/08/2025
#############################################
def main():
    # 1) Ensure CSV exists (run once if needed)
    # data_file_to_csv()

    # 2) Load CSV
    dataset = pd.read_csv(OUTPUT_PATH)

    # 3) Basic stats
    dataset_statistics(dataset)

    # 4) Prepare features/target
    feature_headers = HEADERS[1:-1]   # drop CodeNumber, keep all features
    target_header = HEADERS[-1]      

    # 5) Split
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, feature_headers, target_header)

    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    # 6) Build + Train Pipeline
    pipeline = build_pipeline()
    trained_model = train_pipeline(pipeline, train_x, train_y)
    print("Trained Pipeline :: ", trained_model)

    # 7) Predictions
    predictions = trained_model.predict(test_x)

    # 8) Metrics
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print("Classification Report:\n", classification_report(test_y, predictions))
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions))

    # Feature importances (tree-based)
    plot_feature_importances(trained_model, feature_headers, title="Feature Importances Logistic Regression")

    # 9) Save model (Pipeline) using joblib
    save_model(trained_model, MODEL_PATH)

    # 10) Load model and test a sample
    loaded = load_model(MODEL_PATH)
    sample = test_x.iloc[0]   

#############################################
# Application starter
#############################################
if __name__ == "__main__":
    main()
