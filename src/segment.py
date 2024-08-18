from ultralytics import YOLO
import glob as glob
import numpy as np
import json
import cv2
import os

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
    
def extract_json(img, results, model, save_image=False):
    detections = []
    names = model.names

    if results[0].masks is not None:
        classes = []
        confs = []
        for box in results[0].boxes:
            classes.append(int(box.cls))
            confs.append(float(box.conf))
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
            confidence = confs[i]

            # Prepare detection info
            detection_info = {
                "name": class_name,
                "class": class_id,
                "confidence": confidence,
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

            detections.append(detection_info)

            # Visualize the polygon on the image only if pot hole
            if class_name == "pothole":
                cv2.polylines(img, [polygon], isClosed=True,
                              color=(0, 255, 0), thickness=2)

    # Save the detections to a JSON file
    if save_image:
        return detections, img
    else:
        return detections

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
    
    detections = extract_json(img, results, model)
    # Uncomment to save the image with contours
    # detections, img = extract_json(img, results, model)
    # output_image_path = output_json_path.replace(".json", "_contours.png")
    # cv2.imwrite(output_image_path, img)
    save_detections_as_json(detections, output_json_path)

def segment_yolo(model_path, train_image_dir, test_image_dir):
    """
    Segments images using the YOLO model.
    Args:
        model_path (str): The path to the YOLO model.
        train_image_dir (str): The directory containing the training images.
        test_image_dir (str): The directory containing the test images.
    Returns:
        None
    """
    
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
        test_results = model.predict(
            image_file, iou=0.7, conf=0.35, agnostic_nms=False)
        # Visualize and save detections for training
        output_path = os.path.join(train_output_dir, os.path.basename(
            image_file).replace(".jpg", "_results.json"))
        visualize_and_save_detections(
            image_file, test_results, output_path, model)
    for image_file in test_image_files:
        # Test results for the current image
        test_results = model.predict(
            image_file, iou=0.7, conf=0.35, agnostic_nms=False)
        # Visualize and save detections for training
        output_path = os.path.join(test_output_dir, os.path.basename(
            image_file).replace(".jpg", "_results.json"))
        visualize_and_save_detections(
            image_file, test_results, output_path, model)

def main():
    # Set the paths
    model_path = "data/final_best.pt"
    train_image_dir = "data/train_images/"
    test_image_dir = "data/test_images/"
    
    # Segment the images using the model trained by YoloV8
    segment_yolo(model_path, train_image_dir, test_image_dir)


if __name__ == "__main__":
    main()
