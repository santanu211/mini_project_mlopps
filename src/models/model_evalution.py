import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import json
import pickle  # Added to load the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import dagshub

# Set up DagsHub MLflow tracking
mlflow.set_tracking_uri("https://dagshub.com/santanu211/mini_project_mlopps.mlflow")
dagshub.init(repo_owner='santanu211', repo_name='mini_project_mlopps', mlflow=True)

mlflow.set_experiment("dvc pipeline")

# Load test data
y_test = pd.read_csv(r"C:\Users\Admin\Music\mini_project_mlopps\mini_project\datas\model\actualdt.csv")
y_pred = pd.read_csv(r"C:\Users\Admin\Music\mini_project_mlopps\mini_project\datas\model\predicteddt.csv")

# Load trained model
model_path = r"C:\Users\Admin\Music\mini_project_mlopps\mini_project\models\model.pkl"  # Update path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}

# Start an MLflow run
with mlflow.start_run() as run:
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("auc", auc)

    # Log model parameters if available
    if hasattr(model, "get_params"):
        params = model.get_params()
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

    # Log the trained model
    mlflow.sklearn.log_model(model, "model")

    # Save metrics to a JSON file
    with open("metrics.json", "w") as file:
        json.dump(metrics_dict, file, indent=4)

    # Log the metrics JSON file as an artifact
    mlflow.log_artifact("metrics.json")

    # Save model info
    model_info = {"run_id": run.info.run_id, "model_path": "model"}
    with open("model_info.json", "w") as file:
        json.dump(model_info, file, indent=4)

    # Log the model info file
    mlflow.log_artifact("model_info.json")
