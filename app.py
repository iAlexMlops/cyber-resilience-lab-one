from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        data = np.array(input_data['data']).reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)

        # Prepare data for prediction (assuming time_step=60)
        X = scaled_data[-60:].reshape(1, 60, 1)

        # Make prediction
        prediction = model.predict(X)
        final_prediction = scaler.inverse_transform(prediction)

        return jsonify({"prediction": float(final_prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
