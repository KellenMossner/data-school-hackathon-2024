import json, sys
import pandas as pd
import numpy as np
import logging  # Use the standard logging module
from flask import Flask
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Configure the logger to write to a log file
log_file_handler = logging.FileHandler('logs/model.log', mode='w')
log_file_handler.setLevel(logging.DEBUG)

# Set the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file_handler.setFormatter(formatter)

# Set up the logger
logger = logging.getLogger(__name__)
logger.addHandler(log_file_handler)
logger.setLevel(logging.DEBUG)

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

def train_linear_model(X_train, X_test, y_train, y_test):
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Model trained successfully.")

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model R-squared score on training data: {train_score:.4f}")
    logger.info(f"Model R-squared score on test data: {test_score:.4f}")

    return model

def extract_data(number_of_obs=10):
    # Read training labels using pandas
    df = pd.read_csv('data/train_labels.csv')
    df = df.head(10)  # Limit to the first 10 labels

    # Collect areas for each pothole
    areas = []
    for pothole_number in df['Pothole number']:
        try:
            with open(f'data/train_json/input{pothole_number}.json', 'r') as file:
                json_data = file.read()
            area = calculate_pothole_area(json_data)
            areas.append(area)
        except FileNotFoundError:
            logger.warning(f"JSON file for pothole {pothole_number} not found.")
            areas.append(np.nan)  # Append NaN for missing data
        except Exception as e:
            logger.error(f"Error calculating area for pothole {pothole_number}: {str(e)}")
            areas.append(np.nan)  # Append NaN for error cases

    # Add areas to the dataframe
    df['Area'] = areas

    # Remove rows with NaN values
    df = df.dropna()

    # Prepare data for model
    X = df[['Area']]
    y = df['Bags used ']

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Training labels shape: {y.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the training and test data
    logger.debug(f"X_train:\n{X_train}")
    logger.debug(f"y_train:\n{y_train}")
    logger.debug(f"X_test:\n{X_test}")
    logger.debug(f"y_test:\n{y_test}")

    return X_train, X_test, y_train, y_test

def main():
    try:        
        # Train the model
        print("\nTraining the model...")
        X_train, X_test, y_train, y_test = extract_data()

        lin_model = train_linear_model()
        print("\nModel trained successfully.")
    
    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        logger.error(f"The file '{file_path}' does not contain valid JSON data.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()