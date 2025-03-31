import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml

def load_params(path):
    test_size = yaml.safe_load(open(path, "r"))["data"]["dataload"]["test_size"]
    return test_size


def load_data(data_url: str) -> pd.DataFrame:
    df = pd.read_csv(data_url)
    return df


def preprosed(df: pd.DataFrame) -> pd.DataFrame:
    # 'id' is just an identifier, and 'City', 'Profession', 'Degree' may not be useful
    df = df.drop(columns=['id', 'City', 'Profession', 'Degree'], errors='ignore')

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    # Ensure the directory exists
    raw_data_path = os.path.join(data_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    # Save train and test data
    train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)


def main():
    test_size = load_params("params.yaml")  # Assuming `params.yaml` is in the same directory

    # Check if the dataset exists
    data_path = "datas/Student Depression Dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset file '{data_path}' not found.")
        return
    
    df = load_data(data_path)

    # Preprocess the data
    final_df = preprosed(df)

    # Split the data
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    # Save the processed data
    save_data(train_data, test_data, data_path=r'C:\Users\Admin\Music\mini_project_mlopps\mini_project\datas')


if __name__ == "__main__":
    main()
