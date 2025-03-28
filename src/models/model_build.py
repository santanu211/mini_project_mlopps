import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier




import yaml

def load_param(path):
    """Load hyperparameters from a YAML file."""
    with open(path, "r") as file:
        params = yaml.safe_load(file)
    return params["models"]["model_build"]["n_estimators"], params["models"]["model_build"]["random_state"]


# Load the data
x_train = pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\feature\train_features.csv")
y_train = pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\feature\train_labels.csv")
x_test = pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\feature\test_features.csv")
y_test = pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\feature\test_labels.csv")

# Ensure y_train and y_test are series (not DataFrame)
y_train = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
y_test = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test

# Check if y_train contains continuous values
if y_train.dtype in ['float64', 'float32']:
    print("⚠ Warning: y_train contains continuous values. Converting to categorical bins.")
    
    try:
        y_train = pd.qcut(y_train, q=3, labels=[0, 1, 2], duplicates='drop')  
        y_test = pd.qcut(y_test, q=3, labels=[0, 1, 2], duplicates='drop')  
    except ValueError:
        print("⚠ Error: Too many duplicate values. Using pd.cut instead.")
        y_train = pd.cut(y_train, bins=3, labels=[0, 1, 2])
        y_test = pd.cut(y_test, bins=3, labels=[0, 1, 2])

# Encode labels only if necessary
if y_train.dtype not in ['int64', 'int32']:
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)  # Ensure all labels exist in y_train

# Train the model
def train_model(x_train, y_train):
    """Train a RandomForest model."""
    dt = DecisionTreeClassifier()    
    dt.fit(x_train, y_train)
    return dt

# Predict
def predict(model, x_test):
    return model.predict(x_test)

# Save results
def save_results(y_test, y_pred, data_path):
    data_path = os.path.join(data_path, "model")
    os.makedirs(data_path, exist_ok=True)

    pd.DataFrame(y_test, columns=["Actual"]).to_csv(os.path.join(data_path, "actualdt.csv"), index=False)
    pd.DataFrame(y_pred, columns=["Predicted"]).to_csv(os.path.join(data_path, "predicteddt.csv"), index=False)

# Main execution
def main():
    dt_model = train_model(x_train, y_train)
    y_pred = predict(dt_model, x_test)
    save_results(y_test, y_pred, data_path=r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas")

if __name__ == "__main__":
    main()
