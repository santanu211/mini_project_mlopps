import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data from raw table
train_data = pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\raw\train.csv")
test_data = pd.read_csv("C:/Users/Admin/Music/mlop1/depressed_or_not/datas/raw/test.csv")


def standardize(data):
    scaler = StandardScaler()

    # Convert categorical columns to numeric
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])  # Encode categorical values

    X_scaled = scaler.fit_transform(data)
    return pd.DataFrame(X_scaled, columns=data.columns)  # Return standardized DataFrame

def save(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    # Store the data inside data/processed
    data_path = os.path.join(data_path, "processed")
    os.makedirs(data_path, exist_ok=True)  # Avoid error if directory exists

    train_data.to_csv(os.path.join(data_path, "train.csv"),index=False)
    test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

def main():
    train_prepro_data = standardize(train_data)
    test_prepro_data = standardize(test_data)
    save(train_prepro_data, test_prepro_data, data_path="C:/Users/Admin/Music/mlop1/depressed_or_not/datas")

if __name__ == "__main__":
    main()
