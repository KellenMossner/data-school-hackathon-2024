from flask import Flask, request, jsonify, render_template, send_file
import segment
from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import io
from PIL import Image
from shapely.geometry import Polygon
import math
import base64
import pandas as pd
from predict import (
    calculate_pothole_area,
    calculate_aspect_ratio,
    calculate_length_L2,
    calculate_perimeter,
    calc_max_diameter,
    isL2present,
    pothole_confidence,
    l1_confidence,
    l2_confidence
)

app = Flask(__name__)

# Load your YOLO model
yolo_model = YOLO("data/final_best.pt")

@app.route('/')
def index():
    return render_template('index.html')

# Load your gbm model
with open("data/gbm.pkl", "rb") as f:
    gbm_model = pickle.load(f)

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def extract_features(json_data):
    features = {
        'Pothole number': 0,
        'Area': calculate_pothole_area(json_data),
        'Aspect Ratio': calculate_aspect_ratio(json_data),
        'L2 Length': calculate_length_L2(json_data),
        'Perimeter': calculate_perimeter(json_data),
        'Max Diameter': calc_max_diameter(json_data),
        'Area Squared': np.square(calculate_pothole_area(json_data)),
        'L2 Present': isL2present(json_data),
        'Area_to_Perimeter': calculate_pothole_area(json_data) / calculate_perimeter(json_data),
        'Compactness': 4 * np.pi * calculate_pothole_area(json_data) / (calculate_perimeter(json_data) ** 2),
        'Log_Area': np.log1p(calculate_pothole_area(json_data)),
        'Pothole Confidence': pothole_confidence(json_data),
        'Area_Confidence': calculate_pothole_area(json_data) * pothole_confidence(json_data),
        'L1 Confidence': l1_confidence(json_data),
        'L2 Confidence': l2_confidence(json_data)
    }
    df = pd.DataFrame(features, index=[0])
    return df

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process the image with YOLO
    results = yolo_model.predict(opencv_image, iou=0.7, conf=0.35, agnostic_nms=False)
    detections, img = segment.extract_json(opencv_image, results, yolo_model.model, save_image=True)
    # Extract features from YOLO results
    features = extract_features(detections)
    features['Pothole number'] = int(image_file.filename[1:-4])
    print(features)
    # Make prediction
    cement_bags = max(gbm_model.predict(features)[0], 0.25)
    print(cement_bags)

    # Convert the processed image to a byte stream and encode it in base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "prediction": cement_bags,
        "image": img_base64,
        "message": "Image uploaded and processed successfully!"
    })

if __name__ == '__main__':
    app.run(debug=True)