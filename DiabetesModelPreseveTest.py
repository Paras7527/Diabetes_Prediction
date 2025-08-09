from pathlib import Path    #Accessing path
import joblib               #To preserve the model
import numpy as np


ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "DiabetesPred.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def main():

    pipe = joblib.load(MODEL_PATH)

    sample = np.array([[1,89,66,23,94,28.1,0.167,21]],dtype=float)

    y_predict  = pipe.predict(sample)[0]

    print("Predicted result is :",y_predict)

if __name__ == "__main__":
    main()