from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import glob as glob


def save_detections_as_json(detections, output_json_path):
    """
    Save the detection results as a JSON file.

    Parameters:
        detections (list): List of detection dictionaries containing polygons and metadata.
        output_json_path (str): Path to the output JSON file.
    """
    with open(output_json_path, 'w') as json_file:
        json.dump(detections, json_file, indent=2)
    print(f"Detections saved to {output_json_path}")


def visualize_and_save_detections(image_path, results, output_json_path, model):
    """
    Visualize the segmentation masks on the image and save the masks as a single polygon in a JSON file.
    Merges multiple contours into a single polygon for each pothole.

    Parameters:
        image_path (str): Path to the image file.
        results (YOLO result object): The result object returned by the YOLO model inference.
        output_json_path (str): Path to the output JSON file.
        model (YOLO): The YOLO model object.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Failed to load the image at {os.path.abspath(image_path)}")
        return

    detections = []
    names = model.names

    if results[0].masks is not None:
        classes = []
        for c in results[0].boxes.cls:
            classes.append(int(c))
        for i, mask in enumerate(results[0].masks.data):
            mask = mask.cpu().numpy()  # Convert to numpy array
            # Resize to match image size
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # Find all contours from the mask
            contours, _ = cv2.findContours((mask > 0.5).astype(
                np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Merge all contours into a single polygon
            all_points = np.concatenate(contours)
            hull = cv2.convexHull(all_points)

            # Simplify the polygon to reduce the number of points
            epsilon = 0.001 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # Convert the simplified polygon to a list of points
            polygon = approx.squeeze()

            if polygon.ndim == 1:
                # In case the polygon is a single point
                polygon = polygon[np.newaxis, :]

            # Separate the x and y coordinates
            x_points = polygon[:, 0].tolist()
            y_points = polygon[:, 1].tolist()

            # Get the correct class name using the class id from results[0].boxes.cls
            class_id = classes[i]
            class_name = names[class_id]

            # Prepare detection info
            detection_info = {
                "name": class_name,
                "class": class_id,
                "box": {
                    "x1": float(min(x_points)),
                    "y1": float(min(y_points)),
                    "x2": float(max(x_points)),
                    "y2": float(max(y_points))
                },
                "segments": {
                    "x": x_points,
                    "y": y_points
                }
            }

            # Include confidence if available
            if results[0].probs is not None:
                detection_info["confidence"] = float(results[0].probs[i])

            detections.append(detection_info)

            # Visualize the polygon on the image only if pot hole
            if class_name == "pothole":
                cv2.polylines(img, [polygon], isClosed=True,
                              color=(0, 255, 0), thickness=2)

    # Save the detections to a JSON file
    save_detections_as_json(detections, output_json_path)
    # output_image_path = output_json_path.replace(".json", "_contours.png")
    # cv2.imwrite(output_image_path, img)

def segment_yolo(model_path, train_image_dir, test_image_dir):
    # Load the model
    model = YOLO(model_path)
    train_output_dir = "data/cv_train_out"
    test_output_dir = "data/cv_test_out"
    # Create output directory if it doesn't exist
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    # Get a list of all image files in the directory
    train_image_files = glob.glob(os.path.join(train_image_dir, "*.jpg"))
    test_image_files = glob.glob(os.path.join(test_image_dir, "*.jpg"))

    for image_file in train_image_files:
        # Test results for the current image
        test_results = model(image_file)
        test_results = model.predict(
            image_file, iou=0.7, conf=0.35, agnostic_nms=False)
        # Visualize and save detections for training
        output_path = os.path.join(train_output_dir, os.path.basename(
            image_file).replace(".jpg", "_results.json"))
        visualize_and_save_detections(
            image_file, test_results, output_path, model)
    for image_file in test_image_files:
        # Test results for the current image
        test_results = model(image_file)
        test_results = model.predict(
            image_file, iou=0.7, conf=0.35, agnostic_nms=False)
        # Visualize and save detections for training
        output_path = os.path.join(test_output_dir, os.path.basename(
            image_file).replace(".jpg", "_results.json"))
        visualize_and_save_detections(
            image_file, test_results, output_path, model)

def segment_roboflow(train_image_dir, test_image_dir):
    train_output_dir = "data/robo_cv_train_out"
    test_output_dir = "data/robo_cv_test_out"
    # Create output directory if it doesn't exist
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    API_KEY = os.getenv('API_KEY')
    from roboflow import Roboflow
    rf = Roboflow(api_key=API_KEY) #if this doesnt run then add the api key locally: export API_KEY=your_key
    project = rf.workspace("data-school-hackathon").project("patch-perfect-fq7m5")
    model = project.version(8).model
    
    # Get a list of all image files in the directory
    train_image_files = glob.glob(os.path.join(train_image_dir, "*.jpg"))
    test_image_files = glob.glob(os.path.join(test_image_dir, "*.jpg"))

    for image_file in train_image_files:
        # Test results for the current image
        test_results = model.predict(
            image_file, overlap=30, confidence=0.5)
        # Visualize and save detections for training
        output_path = os.path.join(train_output_dir, os.path.basename(
            image_file).replace(".jpg", "_results.json"))
        save_detections_as_json(test_results.json(), output_path)
    for image_file in test_image_files:
        # Test results for the current image
        test_results = model.predict(
            image_file, overlap=30, confidence=0.5)
        # Visualize and save detections for training
        output_path = os.path.join(test_output_dir, os.path.basename(
            image_file).replace(".jpg", "_results.json"))
        save_detections_as_json(test_results.json(), output_path)

def main():
    # Set the paths
    model_path = "data/kellen_best.pt"
    train_image_dir = "data/train_images/"
    test_image_dir = "data/test_images/"
    
    # Segment the images using the model trained by YoloV8
    # segment_yolo(model_path, train_image_dir, test_image_dir)

    # Segment the images using the model trained by roboflow
    segment_roboflow(train_image_dir, test_image_dir)

if __name__ == "__main__":
    main()
