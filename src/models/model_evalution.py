import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import json

y_test=pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\model\actualdt.csv")
y_pred=pd.read_csv(r"C:\Users\Admin\Music\mlop1\depressed_or_not\datas\model\predicteddt.csv")

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

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)
    