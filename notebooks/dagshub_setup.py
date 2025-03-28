import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/santanu211/mini_project_mlopps.mlflow")

dagshub.init(repo_owner='santanu211', repo_name='mini_project_mlopps', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
