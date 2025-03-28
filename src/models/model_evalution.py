import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import json


import dagshub

mlflow.set_tracking_uri("https://dagshub.com/santanu211/mini_project_mlopps.mlflow")

dagshub.init(repo_owner='santanu211', repo_name='mini_project_mlopps', mlflow=True)

mlflow.set_experiment("dvc pipeline")
y_test=pd.read_csv(r"C:\Users\Admin\Music\mini_project_mlopps\mini_project\datas\model\actualdt.csv")
y_pred=pd.read_csv(r"C:\Users\Admin\Music\mini_project_mlopps\mini_project\datas\model\predicteddt.csv")

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

# Start an MLflow run
with mlflow.start_run():
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("auc", auc)
    
    
    
    # Save metrics to a JSON file
    with open("metrics.json", "w") as file:
        json.dump(metrics_dict, file, indent=4)
    
    # Log the metrics JSON file as an artifact
    mlflow.log_artifact("metrics.json")
    