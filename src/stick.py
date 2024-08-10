import json, sys, csv
from flask import logging
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def calculate_L1_size(json_data):
    data = json.loads(json_data)
    L1 = next(box for box in data['boxes'] if box['label'] == 'L1')
    # Extract the points of the pothole polygon
    points = L1['points']
    
    # Create a Shapely polygon
    poly = Polygon(points)
    
    # Calculate the area of the polygon
    area = poly.area
    
    # Calculate the perimeter of the polygon
    perimeter = poly.length
    
    # Get the bounding box dimensions
    width = float(L1['width'])
    height = float(L1['height'])
    
    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Calculate the percentage of the image occupied by the pothole
    image_width = data['width']
    image_height = data['height']
    image_area = image_width * image_height
    pothole_percentage = (area / image_area) * 100
    
    return {
        'area': area,
        'perimeter': perimeter,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'percentage_of_image': pothole_percentage
    }


def calculate_pothole_area(json_data):
    # Parse the JSON data
    data = json.loads(json_data)
    
    # Find the pothole polygon
    pothole = next(box for box in data['boxes'] if box['label'] == 'pothole')
    
    # Extract the points of the pothole polygon
    points = pothole['points']
    
    # Create a Shapely polygon
    poly = Polygon(points)
    
    # Calculate the area of the polygon
    return poly.area

def train_model():
    # Read training labels
    labels = []
    with open('data/train_labels.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            labels.append((int(row['Pothole number']), float(row['Bags used'])))
    
    # Collect areas for each pothole
    areas = []
    for pothole_number, _ in labels:
        try:
            with open(f'data/{pothole_number}.json', 'r') as file:
                json_data = file.read()
            area = calculate_pothole_area(json_data)
            areas.append([area])
        except FileNotFoundError:
            print(f"Warning: JSON file for pothole {pothole_number} not found.")
    
    # Prepare data for model
    X = np.array(areas)
    y = np.array([bags for _, bags in labels])

    logging.info(f"Training data shape: {X.shape}")
    logging.info(f"Training labels shape: {y.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model R-squared score on training data: {train_score:.4f}")
    print(f"Model R-squared score on test data: {test_score:.4f}")
    
    return model

def main():
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide the path to the JSON file as a command-line argument.")
        sys.exit(1)
    
    # Get the file path from the command-line argument
    file_path = sys.argv[1]
    
    try:
        # Read the JSON data from the file
        with open(file_path, 'r') as file:
            json_data = file.read()
        
        # Calculate the pothole area
        area = calculate_pothole_area(json_data)
        
        print(f"Pothole Area: {area:.2f}")
        
        # Train the model
        print("\nTraining the model...")
        model = train_model()
        # Use the model to predict bags for the current pothole
        predicted_bags = model.predict([[area]])[0]
        
        print(f"\nPredicted number of bags: {predicted_bags:.2f}")
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' does not contain valid JSON data.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()