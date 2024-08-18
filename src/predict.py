from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from shapely.geometry import Polygon
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging
import math
import json
import os

# Configure logging
open('logs/model.log', 'w').close()
logging.basicConfig(filename='logs/model.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_pothole_area(json_data):
    """
    Calculate the area of a pothole based on the provided JSON data.
    Parameters:
    json_data (list): A list of dictionaries containing JSON data.
    Returns:
    float or None: The calculated area of the pothole, multiplied by the square of the ratio if available. Returns None if no pothole is found.
    """
    
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

def pothole_confidence(json_data):
    """
    Calculates the confidence level of a pothole in the given JSON data.
    Parameters:
    json_data (list): A list of dictionaries representing JSON data.
    Returns:
    float: The confidence level of the pothole.
    """
    
    for item in json_data:
        if item['name'] == 'pothole':
            return item['confidence']

def l1_confidence(json_data):
    """
    Calculates the confidence level of the L1 segment in the given JSON data.
    Parameters:
    json_data (list): A list of dictionaries representing JSON data.
    Returns:
    float: The confidence level of the L1 segment.
    """
    
    for item in json_data:
        if item['name'] == 'L1':
            return item['confidence']

def l2_confidence(json_data):
    """
    Calculates the confidence level of the L2 segment in the given JSON data.
    Parameters:
    json_data (list): A list of dictionaries representing JSON data.
    Returns:
    float: The confidence level of the L2 segment.
    """
    
    for item in json_data:
        if item['name'] == 'L2':
            return item['confidence']

def distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    Parameters:
    point1 (tuple): The coordinates of the first point in the form (x, y).
    point2 (tuple): The coordinates of the second point in the form (x, y).
    Returns:
    float: The Euclidean distance between the two points.
    """
    
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_length_L1(json_data):
    """
    Calculate the length of the L1 segment in the given JSON data.
    Parameters:
    - json_data (list): A list of dictionaries representing JSON data.
    Returns:
    - float or None: The length of the L1 segment if found, None otherwise.
    """
    # Rest of the code...
    
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
    """
    Calculate the L1 ratio based on the given JSON data.
    Parameters:
    json_data (dict): A dictionary containing the JSON data.
    Returns:
    float or None: The calculated L1 ratio if the length of L1 is not None, otherwise None.
    """
    
    DIAGONAL_L1 = 503.5
    l1_length = calculate_length_L1(json_data)
    if l1_length is not None:
        return DIAGONAL_L1/l1_length
    else:
        return None

def calculate_length_L2(json_data):
    """
    Calculates the length of the L2 segment in the given JSON data.
    Parameters:
    - json_data (list): A list of dictionaries representing JSON data.
    Returns:
    - float: The length of the L2 segment.
    Example:
    >>> json_data = [
    ...     {
    ...         'name': 'L1',
    ...         'segments': {
    ...             'x': [1, 2, 3],
    ...             'y': [4, 5, 6]
    ...         }
    ...     },
    ...     {
    ...         'name': 'L2',
    ...         'segments': {
    ...             'x': [7, 8, 9],
    ...             'y': [10, 11, 12]
    ...         }
    ...     }
    ... ]
    >>> calculate_length_L2(json_data)
    5.196152422706632
    """
    
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
    """
    Calculate the aspect ratio of the pothole from the given JSON data.
    Parameters:
    - json_data (list): A list of dictionaries containing information about the pothole.
    Returns:
    - float: The aspect ratio of the pothole.
    Notes:
    - The aspect ratio is calculated by dividing the width of the pothole by its height.
    - If no pothole is found in the JSON data, a default aspect ratio of 0.9 is returned.
    """
    
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
    """
    Calculate the perimeter of a pothole based on the given JSON data.
    Parameters:
    json_data (list): A list of dictionaries representing the JSON data.
    Returns:
    float: The perimeter of the pothole.
    """
    # Rest of the code...
    
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
    """
    Calculates the maximum diameter of a pothole from the given JSON data.
    Parameters:
    json_data (list): A list of dictionaries representing the JSON data.
    Returns:
    float: The maximum diameter of a pothole.
    """
    # Rest of the code...
    
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
    """
    Checks if the given JSON data contains an item with the name 'L2'.
    Parameters:
    - json_data (list): A list of dictionaries representing JSON data.
    Returns:
    - bool: True if an item with the name 'L2' is present in the JSON data, False otherwise.
    """
    
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
    pothole_confidences = []
    l1_confidences = []
    l2_confidences = []

    for _, row in df.iterrows():
        image_name = row['Pothole number']

        image_file_path = os.path.join(
            json_dir, f'p{int(image_name)}_results.json')
        logging.debug(f"Processing image: {image_file_path}")

        # Check if file exists
        if not os.path.isfile(image_file_path):
            logging.debug(f"JSON file not found for image: {image_name}. Skipping this image.")
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
            
            # pothole confidence
            pothole_conf = pothole_confidence(json_data)
            pothole_confidences.append(pothole_conf)
            
            # L1 confidence
            l1_conf = l1_confidence(json_data)
            l1_confidences.append(l1_conf)
            
            # L2 confidence
            l2_conf = l2_confidence(json_data)
            l2_confidences.append(l2_conf)

            # Keep the row only if it has a valid JSON file
            valid_rows.append(row)

            # Log the area
            logging.debug(f"Area of pothole in image {image_name}: {area:.2f}")
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in file: {image_file_path}")
        except Exception as e:
            logging.error(f"Error processing {image_file_path}: {str(e)}")

    # Create a new DataFrame with only valid rows and feature engineering
    valid_df = pd.DataFrame(valid_rows)
    valid_df['Area'] = areas
    valid_df['Aspect Ratio'] = aspect_ratios
    valid_df['L2 Length'] = l2_lengths
    valid_df['Perimeter'] = perimeters
    valid_df['Max Diameter'] = max_diameters
    valid_df['Area Squared'] = np.square(valid_df['Area'])
    valid_df['L2 Present'] = l2_presents
    valid_df['Area_to_Perimeter'] = valid_df['Area'] / valid_df['Perimeter']
    valid_df['Compactness'] = 4 * np.pi * valid_df['Area'] / (valid_df['Perimeter'] ** 2)
    valid_df['Log_Area'] = np.log1p(valid_df['Area'])
    valid_df['Pothole Confidence'] = pothole_confidences
    valid_df['Area_Confidence'] = valid_df['Area'] * valid_df['Pothole Confidence']
    valid_df['L1 Confidence'] = l1_confidences
    valid_df['L2 Confidence'] = l2_confidences
    return valid_df

def train_linear_model(X, y):
    """
    Trains a linear regression model using the given features (X) and target variable (y).
    Parameters:
    - X (array-like): The features used for training the model.
    - y (array-like): The target variable used for training the model.
    Returns:
    - model: The trained linear regression model.
    This function splits the data into training and test sets, fits a linear regression model on the training data,
    and evaluates the model's performance on the test data. It also performs some additional steps such as making predictions,
    adjusting predictions if they are less than 0.25, calculating the R-squared score, and logging the results.
    Note: This function assumes that the necessary libraries (e.g., train_test_split, LinearRegression, logging, np, pd)
    have been imported before calling this function.
    """
    # Function code here
    pass
    
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

def improved_model(X, y):
    """
    Trains an improved regression model using feature engineering, ensemble learning, and hyperparameter tuning.
    Parameters:
    X (array-like): The input features.
    y (array-like): The target variable.
    Returns:
    object: The best trained regression model.
    """

    # Create base models
    linear_reg = LinearRegression()
    ridge_reg = Ridge()
    lasso_reg = Lasso()
    rf_reg = RandomForestRegressor(random_state=42)

    # Create voting regressor
    voting_reg = VotingRegressor([
        ('linear', linear_reg),
        ('ridge', ridge_reg),
        ('lasso', lasso_reg),
        ('rf', rf_reg)
    ])

    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', voting_reg)
    ])

    # Define hyperparameters to tune
    param_grid = {
        'regressor__linear__fit_intercept': [True, False],
        'regressor__ridge__alpha': [0.1, 1.0, 10.0],
        'regressor__ridge__fit_intercept': [True, False],
        'regressor__lasso__alpha': [0.1, 1.0, 10.0],
        'regressor__lasso__fit_intercept': [True, False],
        'regressor__rf__n_estimators': [100, 200],
        'regressor__rf__max_depth': [None, 10, 20],
        'regressor__rf__min_samples_split': [2, 5]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    # Get best model
    best_model = grid_search.best_estimator_

    # Print best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best RMSE:", np.sqrt(-grid_search.best_score_))

    return best_model

def grid_search(X, y):
    """
    Perform grid search to estimate the best parameters for the gradient boosting model.
    Parameters:
    - X (array-like): The input features.
    - y (array-like): The target variable.
    Returns:
    - best_params (dict): The best parameters found during grid search.
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5]
    }
    
    # Create the gradient boosting model
    gbm = GradientBoostingRegressor(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    logging.info(f"Best model parameters: {best_params}")
    print(f"Best model parameters: {best_params}")
    
    return best_params

def gradient_boosting_model(X, y):
    """
    Trains a gradient boosting model on the given data and returns the trained model.
    Parameters:
    X (array-like): The input features.
    y (array-like): The target variable.
    Returns:
    object: The trained gradient boosting model.
    Raises:
    None
    Example:
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    y = [10, 11, 12]
    model = gradient_boosting_model(X, y)
    """
    
    # Define the model: Best results with n_estimators=300, learning_rate=0.1, max_depth=3
    gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.2, max_depth=3, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    gbm.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = gbm.predict(X_test)
    
    # Ensure predictions are not less than 0.25
    y_pred = np.maximum(y_pred, 0.25)
    
    # Calculate and log R-squared
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    logging.info(f"Gradient Boosting - R-squared: {r2:.4f}")
    logging.info(f"Gradient Boosting - Mean Squared Error: {mse:.4f}")
    
    print(f"R-squared on test data: {r2:.4f}")
    print(f"Mean Squared Error on test data: {mse:.4f}")
    
    return gbm

def cross_validation(X, y, n_splits=10):
    """
    Perform cross-validation using KFold and RandomForestRegressor.
    Parameters:
    - X (pandas.DataFrame): The input features.
    - y (pandas.Series): The target variable.
    - n_splits (int): The number of folds for cross-validation. Default is 10.
    Returns:
    - mean_r2 (float): The mean R-squared score across all folds.
    - std_r2 (float): The standard deviation of the R-squared scores across all folds.
    """
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.2, max_depth=3, random_state=42)

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
    print(f"CV Mean R-squared: {mean_r2:.4f} (+/- {std_r2:.4f})")
    
    return mean_r2, std_r2

def main():
    json_dir = 'data/cv_train_out'
    csv_file = 'data/train_labels.csv'

    try:
        # ----- DATA PREPROCESSING -----
        df_train = extract_data(json_dir, csv_file)
        print("Data extracted successfully")
        df_train = df_train.dropna()

        X = df_train.loc[:, df_train.columns != 'Bags used '].copy()
        y = df_train['Bags used ']

        # Save processed data to a CSV file
        df_train.to_csv('data/processed_data.csv', index=False)
        
        # Grid search for best model parameters
        # Note: best parameters did not return best score on kaggle
        # model_params = grid_search(X, y)
        
        # ----- IMPROVED MODEL -----
        # model = improved_model(X, y)
        model = gradient_boosting_model(X, y)

        # ----- CROSS-VALIDATION -----
        mean_r2, std_r2 = cross_validation(X, y)

    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")

    try:
        # ------------------- Test Model -------------------------------------------
        df_test = extract_data('data/cv_test_out', 'data/test_labels.csv')

        df_test['Area'] = df_test['Area'].fillna(0)

        # Save processed data to a CSV file
        df_test.to_csv('data/processed_test_data.csv', index=False)

        X_test = df_test.loc[:, df_test.columns != 'Bags used '].copy()
        # Make predictions using the improved model
        y_pred = model.predict(X_test)

        # Ensure predictions are not less than 0.25
        y_pred = np.maximum(y_pred, 0.25)

        df_test['Bags used '] = y_pred
        df_test['Pothole number'] = df_test['Pothole number'].astype(int)
        df_test[['Pothole number', 'Bags used ']].to_csv(
            'data/test_results.csv', index=False)
        print(df_test)
        logging.info("Results saved to data/test_results.csv")

    except Exception as e:
        logging.error(f"An error occurred in test execution: {str(e)}")

if __name__ == "__main__":
    main()
