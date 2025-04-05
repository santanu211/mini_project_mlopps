from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import mlflow

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Santanu211"
repo_name = "mini_project_mlopps"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Initialize Flask app
app = Flask(__name__)

# load model from model registry
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

model_path = pickle.load(open('models/model.pkl','rb'))





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure all required features are provided
        features = []
        for i in range(1, 14):
            value = request.form.get(f'feature{i}', '').strip()
            if value == '':
                return jsonify({'error': f'Missing value for feature{i}'})
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({'error': f'Invalid input for feature{i}. Must be a number.'})

        # Convert features into NumPy array
        features_array = np.array([features])

        # Make prediction
        prediction = model.predict(features_array)[0]
        result = "Depression Detected" if prediction == 1 else "No Depression"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
