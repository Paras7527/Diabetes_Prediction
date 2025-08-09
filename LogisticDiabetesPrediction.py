from pathlib import Path    #Accessing path
import joblib           

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "DiabetesPred.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def Diabetes_Prediction(Datapath):
    df = pd.read_csv(Datapath)

    print("Data Loaded Successfully!!!")
    print("The Initial data in the dataset : \n",df.head())
    print("The Shape of the Data is :",df.shape)

    print("The Description of the Dataset : \n",df.describe())

    X = df.drop(columns=['Outcome'])
    Y = df['Outcome']

    scaled = StandardScaler()
    x_scale = scaled.fit_transform(X)

    x_train,x_test,y_train,y_test = train_test_split(x_scale,Y,test_size=TEST_SIZE,random_state=RANDOM_STATE)

    Pipe = Pipeline([
        ("scaler",StandardScaler()),
        ("clf",LogisticRegression(max_iter=100))
    ])

    Pipe.fit(x_train,y_train)
    y_predict = Pipe.predict(x_test)

    print("Accuracy is : ",accuracy_score(y_test,y_predict))

    joblib.dump(Pipe,MODEL_PATH)

def main():
    Diabetes_Prediction("diabetes.csv")

if __name__ == "__main__":
    main()