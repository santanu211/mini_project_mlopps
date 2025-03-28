#imporrt data from procedssed data
import numpy as np
import pandas as pd
import os

# Import data from processed data
train_data = pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\processed\train.csv")
test_data = pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\processed\test.csv")

# Fill missing values
train_data.fillna(0, inplace=True)  # Assuming depression is numerical
test_data.fillna(0, inplace=True)

def sep_lab(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separates features and labels from the dataset."""
    x = df.drop(columns=['Depression'])
    y = df["Depression"]
    return x, y

def save(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series, data_path: str) -> None:
    """Saves train and test features and labels to CSV files."""
    data_path = os.path.join(data_path, "feature")
    os.makedirs(data_path, exist_ok=True)

    train_x.to_csv(os.path.join(data_path, "train_features.csv"), index=False)
    train_y.to_csv(os.path.join(data_path, "train_labels.csv"), index=False)

    test_x.to_csv(os.path.join(data_path, "test_features.csv"), index=False)
    test_y.to_csv(os.path.join(data_path, "test_labels.csv"), index=False)

def main():
    train_x, train_y = sep_lab(train_data)
    test_x, test_y = sep_lab(test_data)
    
    save(train_x, train_y, test_x, test_y, data_path="C:/Users/Admin/Music/mlop1/depressed_or_not/datas")

if __name__ == "__main__":
    main()

