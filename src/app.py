from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import io
from PIL import Image
from shapely.geometry import Polygon
import math

app = Flask(__name__)

# Load your YOLO model
yolo_model = YOLO("data/final_best.pt")

# Load your gbm model
with open("data/gbm.pkl", "rb") as f:
    gbm_model = pickle.load(f)

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def extract_features(json_data):
    features = {}
    
    # Calculate area
    for item in json_data:
        if item['name'] == 'pothole':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            poly = Polygon(points)
            features['Area'] = poly.area
            
            # Calculate aspect ratio
            x1, y1, x2, y2 = item['box']['x1'], item['box']['y1'], item['box']['x2'], item['box']['y2']
            width, height = x2 - x1, y2 - y1
            features['Aspect Ratio'] = width / height
            
            # Calculate perimeter
            perimeter = sum(distance(points[i], points[i+1]) for i in range(len(points)-1))
            perimeter += distance(points[-1], points[0])
            features['Perimeter'] = perimeter
            
            # Calculate max diameter
            max_diameter = max(distance(points[i], points[j]) for i in range(len(points)) for j in range(i+1, len(points)))
            features['Max Diameter'] = max_diameter
            
            features['Area Squared'] = features['Area'] ** 2
            features['Area_to_Perimeter'] = features['Area'] / features['Perimeter']
            features['Compactness'] = 4 * np.pi * features['Area'] / (features['Perimeter'] ** 2)
            features['Log_Area'] = np.log1p(features['Area'])
            features['Pothole Confidence'] = item['confidence']
            features['Area_Confidence'] = features['Area'] * features['Pothole Confidence']
            
            break
    
    # Add L1 and L2 related features
    for item in json_data:
        if item['name'] == 'L1':
            features['L1 Confidence'] = item['confidence']
        elif item['name'] == 'L2':
            features['L2 Present'] = 1
            features['L2 Confidence'] = item['confidence']
            points = list(zip(item['segments']['x'], item['segments']['y']))
            features['L2 Length'] = max(distance(points[i], points[j]) for i in range(len(points)) for j in range(i+1, len(points)))
    
    if 'L2 Present' not in features:
        features['L2 Present'] = 0
        features['L2 Confidence'] = 0
        features['L2 Length'] = 0
    
    return features

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

    # Extract features from YOLO results
    json_data = results[0].tojson()
    features = extract_features(json_data)

    # Prepare features for prediction
    feature_vector = [features[col] for col in ['Area', 'Aspect Ratio', 'L2 Length', 'Perimeter', 'Max Diameter', 
                                                'Area Squared', 'L2 Present', 'Area_to_Perimeter', 'Compactness', 
                                                'Log_Area', 'Pothole Confidence', 'Area_Confidence', 'L1 Confidence', 
                                                'L2 Confidence']]

    # Make prediction
    cement_bags = max(gbm_model.predict([feature_vector])[0], 0.25)

    return jsonify({
        "cement_bags": cement_bags,
        "features": features
    })

if __name__ == '__main__':
    app.run(debug=True)