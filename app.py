from flask import Flask, request, jsonify
import cv2
import numpy as np
from joblib import load
from main import predict_face

app = Flask(__name__)

import logging

# Load the trained models
dt = load('dt_lbp_hog.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Use the predict_face function
    dt_pred = predict_face(img, dt)

    logging.info('Received and processed an image')

    if dt_pred is not None:
        return jsonify({
            'dt_pred': int(dt_pred[0]),
        })
    else:
        return jsonify({
            'error': 'No faces detected',
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

