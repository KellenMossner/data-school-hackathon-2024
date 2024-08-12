import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
import logging
import math

# Configure logging
open('logs/model.log', 'w').close()
logging.basicConfig(filename='logs/model.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_pothole_area(json_data):
    ratio = calculate_L1_ratio(json_data)
    for item in json_data:
        if item['name'] == 'pothole':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            poly = Polygon(points)
            if ratio is not None:
                return poly.area
            else:
                return None
    return None


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_length_L1(json_data):
    for item in json_data:
        if item['name'] == 'L1':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            max_distance = 0

            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = distance(points[i], points[j])
                    if dist > max_distance:
                        max_distance = dist
            return max_distance

    return None


def calculate_L1_ratio(json_data):
    DIAGONAL_L1 = 503.5
    l1_length = calculate_length_L1(json_data)
    if l1_length is not None:
        return l1_length / DIAGONAL_L1
    else:
        return None


def extract_data(json_dir, csv_file):
    json_dir = os.path.abspath(json_dir)
    csv_file = os.path.abspath(csv_file)

    logging.info(f"Using JSON directory: {json_dir}")
    logging.info(f"Using CSV file: {csv_file}")

    df = pd.read_csv(csv_file)
    valid_rows = []  # Store rows that have valid JSON files
    areas = []

    for _, row in df.iterrows():
        image_name = row['Pothole number']

        image_file_path = os.path.join(
            json_dir, f'p{int(image_name)}_results.json')
        logging.debug(f"Processing image: {image_file_path}")

        # Check if file exists
        if not os.path.isfile(image_file_path):
            logging.debug(f"JSON file not found for image: {
                          image_name}. Skipping this image.")
            continue  # Skip to the next iteration if the JSON file is not found

        try:
            with open(image_file_path, 'r') as file:
                json_data = json.load(file)
            area = calculate_pothole_area(json_data)
            areas.append(area)
            # Keep the row only if it has a valid JSON file
            valid_rows.append(row)

            # Log the area
            logging.debug(f"Area of pothole in image {image_name}: {area:.2f}")
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in file: {image_file_path}")
        except Exception as e:
            logging.error(f"Error processing {image_file_path}: {str(e)}")

    # Create a new DataFrame with only valid rows
    valid_df = pd.DataFrame(valid_rows)
    valid_df['Area'] = areas

    return valid_df


def train_linear_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    logging.info(f"Model R-squared score on training data: {train_score:.4f}")
    logging.info(f"Model R-squared score on test data: {test_score:.4f}")

    return model, X_test, y_test


def train_jonty_model(X, y):
    pass


def train_kellen_model(X, y):
    pass


def main():
    json_dir = 'data/cv_train_out'
    csv_file = 'data/train_labels.csv'

    try:
        print("Extracting data...")
        df = extract_data(json_dir, csv_file)
        print("Data extracted successfully.")
        logging.info(f"Processed data shape: {df.shape}")
        logging.debug(f"Processed data columns: {df.columns}")
        logging.debug(f"Dataframe: {df.head()}")
        # Drop NA
        df = df.dropna()

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
