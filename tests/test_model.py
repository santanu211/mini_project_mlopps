import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            dagshub_token = os.getenv("DAGSHUB_PAT")
            if not dagshub_token:
                raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            dagshub_url = "https://dagshub.com"
            repo_owner = "Santanu211"
            repo_name = "mini_project_mlopps"

            mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

            cls.new_model_name = "my_model"
            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
            
            if cls.new_model_version is None:
                raise ValueError(f"No model version found for {cls.new_model_name}")

            cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except Exception as e:
            raise RuntimeError(f"Error in model loading: {str(e)}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        try:
            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=[stage])
            return latest_version[0].version if latest_version else None
        except Exception as e:
            raise RuntimeError(f"Error retrieving latest model version: {str(e)}")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model, "Model failed to load")

    def test_model_signature(self):
        try:
            sample_data = {
                'Gender': ['Male'],
                'Age': [21],
                'Academic Pressure': [3],
                'Work Pressure': [2],
                'CGPA': [3.5],
                'Study Satisfaction': [4],
                'Job Satisfaction': [3],
                'Sleep Duration': [6],
                'Dietary Habits': [2],
                'Have you ever had suicidal thoughts ?': ['No'],
                'Work/Study Hours': [5],
                'Financial Stress': [3],
                'Family History of Mental Illness': ['No'],
            }
            sample_df = pd.DataFrame(sample_data)

            if not hasattr(self, 'holdout_data'):
                raise AttributeError("holdout_data attribute is missing")

            sample_df = sample_df.reindex(columns=self.holdout_data.columns[:-1], fill_value=0)
            prediction = self.new_model.predict(sample_df)

            self.assertEqual(sample_df.shape[1], self.holdout_data.shape[1] - 1)
            self.assertEqual(len(prediction), sample_df.shape[0])
            self.assertEqual(len(prediction.shape), 1)
        except Exception as e:
            self.fail(f"Model signature test failed: {str(e)}")

    def test_model_performance(self):
        try:
            if not hasattr(self, 'holdout_data'):
                raise AttributeError("holdout_data attribute is missing")

            X_holdout = self.holdout_data.iloc[:, 0:-1]
            y_holdout = self.holdout_data.iloc[:, -1]
            
            y_pred_new = self.new_model.predict(X_holdout)
            
            accuracy_new = accuracy_score(y_holdout, y_pred_new)
            precision_new = precision_score(y_holdout, y_pred_new, zero_division=0)
            recall_new = recall_score(y_holdout, y_pred_new, zero_division=0)
            f1_new = f1_score(y_holdout, y_pred_new, zero_division=0)
            roc_auc_new = roc_auc_score(y_holdout, y_pred_new)

            expected_thresholds = {
                "accuracy": 0.40,
                "precision": 0.40,
                "recall": 0.40,
                "f1": 0.40
            }

            self.assertGreaterEqual(accuracy_new, expected_thresholds["accuracy"], "Accuracy below threshold")
            self.assertGreaterEqual(precision_new, expected_thresholds["precision"], "Precision below threshold")
            self.assertGreaterEqual(recall_new, expected_thresholds["recall"], "Recall below threshold")
            self.assertGreaterEqual(f1_new, expected_thresholds["f1"], "F1 score below threshold")
        except Exception as e:
            self.fail(f"Model performance test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()
