import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import logging
import math
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector


# Configure logging
open('logs/model.log', 'w').close()
logging.basicConfig(filename='logs/model.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_pothole_area(json_data):
    ratio = calculate_L1_ratio(json_data)
    for item in json_data:
        if item['name'] == 'pothole':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            if ratio is not None:
                poly = Polygon(points)
                return poly.area*ratio**2
            else:
                poly = Polygon(points)
                ratio = 503.5/calculate_length_L2(json_data)
                return poly.area*ratio**2
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
        return DIAGONAL_L1/l1_length
    else:
        return None


def calculate_length_L2(json_data):
    for item in json_data:
        if item['name'] == 'L2':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            max_distance = 0

            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = distance(points[i], points[j])
                    if dist > max_distance:
                        max_distance = dist
            return max_distance

    return 121


def calculate_aspect_ratio(json_data):
    # Calculate aspect ratio of the pothole from boxes
    for item in json_data:
        if item['name'] == 'pothole':
            x1 = item['box']['x1']
            x2 = item['box']['x2']
            y1 = item['box']['y1']
            y2 = item['box']['y2']
            width = x2 - x1
            height = y2 - y1
            return width / height
    return 0.9


def calculate_perimeter(json_data):
    for item in json_data:
        if item['name'] == 'pothole':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            perimeter = 0
            for i in range(len(points)):
                if i == len(points) - 1:
                    perimeter += distance(points[i], points[0])
                else:
                    perimeter += distance(points[i], points[i + 1])
            return perimeter
    return 763.7


def calc_max_diameter(json_data):
    for item in json_data:
        if item['name'] == 'pothole':
            points = list(zip(item['segments']['x'], item['segments']['y']))
            max_distance = 0

            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = distance(points[i], points[j])
                    if dist > max_distance:
                        max_distance = dist
            return max_distance
    return 289

def isL2present(json_data):
    for item in json_data:
        if item['name'] == 'L2':
            return True
    return False

def extract_data(json_dir, csv_file):
    json_dir = os.path.abspath(json_dir)
    csv_file = os.path.abspath(csv_file)

    logging.info(f"Using JSON directory: {json_dir}")
    logging.info(f"Using CSV file: {csv_file}")

    df = pd.read_csv(csv_file)
    valid_rows = []  # Store rows that have valid JSON files
    areas = []
    aspect_ratios = []
    l2_lengths = []
    perimeters = []
    max_diameters = []
    l2_presents = []

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

            # Add aspect ratio predictor
            aspect_ratio = calculate_aspect_ratio(json_data)
            aspect_ratios.append(aspect_ratio)

            # Add L2 length predictor
            l2_length = calculate_length_L2(json_data)
            l2_lengths.append(l2_length)

            # Add perimeter predictor
            perimeter = calculate_perimeter(json_data)
            perimeters.append(perimeter)

            # Max diameter
            max_diameter = calc_max_diameter(json_data)
            max_diameters.append(max_diameter)

            # L2 present
            l2_present = isL2present(json_data)
            l2_presents.append(l2_present)

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
    valid_df['Aspect Ratio'] = aspect_ratios
    valid_df['L2 Length'] = l2_lengths
    valid_df['Perimeter'] = perimeters
    valid_df['Max Diameter'] = max_diameters
    valid_df['Area Squared'] = np.square(valid_df['Area'])
    valid_df['L2 Present'] = l2_presents
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
    # Make predictions on test set
    y_pred = model.predict(X_test)

    # # Log some sample predictions
    # for true, pred in zip(y_test[:5], y_pred[:5]):
    #     logging.info(f"True: {true}, Predicted: {pred:.2f}")

    # if the prediction is less than 0.25, set to 0.25
    y_pred = np.where(y_pred < 0.25, 0.25, y_pred)

    # Get r-squared of test data
    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)

    compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    logging.info(f"Predicted bags used vs actual: \n{compare}")

    print(f"R-squared on test data: {r2:.4f}")

    # print summary
    lm_model_summary = pd.DataFrame(
        model.coef_, X.columns, columns=['Coefficient'])
    logging.info("Model Coefficients:")
    logging.info(lm_model_summary)
    logging.info("Model training completed successfully.")
    return model


def stepwise_selection(X, y, forward=True):
    logging.info("Performing stepwise selection...")
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    sfs = SequentialFeatureSelector(
        model, forward=forward, k_features='best', scoring='r2', cv=10)
    sfs_fit = sfs.fit(X_train, y_train)

    selected_features = sfs_fit.k_feature_names_
    logging.info(f"Selected features: {selected_features}")

    X_train_sfs = sfs_fit.transform(X_train)
    X_test_sfs = X_test[selected_features].copy()

    model.fit(X_train_sfs, y_train)
    y_pred = model.predict(X_test_sfs)

    # Evaluate performance
    r2_score = r2_score(y_test, y_pred)
    logging.info("R-squared score with selected features:", r2_score)
    return model, selected_features


def perform_cross_validation(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = LinearRegression()

    cv_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        cv_scores.append(score)

        logging.info(f"Fold {fold} R-squared: {score:.4f}")

    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)

    logging.info(f"Cross-validation results:")
    logging.info(f"Mean R-squared: {mean_r2:.4f} (+/- {std_r2:.4f})")

    return mean_r2, std_r2


def main():
    json_dir = 'data/cv_train_out'
    csv_file = 'data/train_labels.csv'

    try:
        # ----- DATA PREPROCESSING -----
        df_train = extract_data(json_dir, csv_file)
        print("Data extracted successfully")
        df_train = df_train.dropna()

        X = df_train[['Area', 'Aspect Ratio',
                      'Perimeter', 'Max Diameter', 'Area Squared', 'L2 Present', 'L2 Length']].copy()
        y = df_train['Bags used ']

        logging.info(f"Processed data shape: {df_train.shape}")
        logging.debug(f"Processed data columns: {df_train.columns}")
        logging.debug(f"Dataframe: {df_train.head()}")

        # Save processed data to a CSV file
        df_train.to_csv('data/processed_data.csv', index=False)

        # print average Aspect Ratio and L2 Length
        # logging.info(f"Average Aspect Ratio: {X['Aspect Ratio'].mean()}")
        # logging.info(f"Average Diameter: {X['Max Diameter'].mean()}")

        # ----- SIMPLE LINEAR REGRESSION MODEL -----
        lm_model = train_linear_model(X, y)

        # ----- CROSS-VALIDATION -----
        mean_r2, std_r2 = perform_cross_validation(X, y)

        # ----- FEATURE SELECTION -----
        # stepwise_model, selected_features = stepwise_selection(X, y, forward=True)

    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")

    try:
        # ------------------- Test Model -------------------------------------------
        # Print csv file
        df_test = extract_data('data/cv_test_out', 'data/test_labels.csv')

        df_test['Area'] = df_test['Area'].fillna(0)

        # Save processed data to a CSV file
        df_test.to_csv('data/processed_test_data.csv', index=False)

        X_test = df_test[['Area', 'Aspect Ratio',
                          'Perimeter', 'Max Diameter', 'Area Squared', 'L2 Present', 'L2 Length']].copy()
        y_test = df_test['Bags used ']
        y_pred = np.round(np.abs(lm_model.predict(X_test)), 2)
        y_pred = np.where(y_pred < 0.25, 0.25, y_pred)

        df_test['Bags used '] = y_pred
        df_test['Pothole number'] = df_test['Pothole number'].astype(int)
        df_test[['Pothole number', 'Bags used ']].to_csv(
            'data/test_results.csv', index=False)
        print(df_test)
        logging.info("Results saved to data/test_results.csv")

        # ------------------- Test Model -------------------------------------------

    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")


if __name__ == "__main__":
    main()
