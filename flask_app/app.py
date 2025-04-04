from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Load Model
model_path = 'models/model.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
