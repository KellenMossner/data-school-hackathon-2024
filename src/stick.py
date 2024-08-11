import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
import logging

# Configure logging
open('logs/model.log', 'w').close()
logging.basicConfig(filename='logs/model.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_pothole_area(json_data):
    for item in json_data:
        if item['name'] == 'pothole':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            poly = Polygon(points)
            return poly.area
    return None

def extract_data(json_dir, csv_file):
    df = pd.read_csv(csv_file)
    areas = []

    for _, row in df.iterrows():
        image_name = row['Image']
        logging.info(f"Processing image: {image_name}")
        json_file = os.path.join(json_dir, f'cv_train_{image_name}.json')
        
        try:
            with open(json_file, 'r') as file:
                json_data = json.load(file)
            area = calculate_pothole_area(json_data)
            areas.append(area)
        except FileNotFoundError:
            logging.warning(f"JSON file not found for image: {image_name}")
            areas.append(None)
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in file: {json_file}")
            areas.append(None)
        except Exception as e:
            logging.error(f"Error processing {json_file}: {str(e)}")
            areas.append(None)

    df['Area'] = areas
    df = df.dropna()
    return df

def train_linear_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logging.info(f"Model R-squared score on training data: {train_score:.4f}")
    logging.info(f"Model R-squared score on test data: {test_score:.4f}")
    
    return model, X_test, y_test

def main():
    json_dir = 'data/cv_train_out'
    csv_file = 'data/train_labels.csv'
    
    try:
        print("Extracting data...")
        df = extract_data(json_dir, csv_file)
        print("Data extracted successfully.")
        logging.info(f"Processed data shape: {df.shape}")
        
        X = df[['Area']]
        y = df['Bags used ']
        
        model, X_test, y_test = train_linear_model(X, y)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Log some sample predictions
        for true, pred in zip(y_test[:5], y_pred[:5]):
            logging.info(f"True: {true}, Predicted: {pred:.2f}")
        
        logging.info("Model training completed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")

if __name__ == "__main__":
    main()