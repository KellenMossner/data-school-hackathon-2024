from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime

def predict_all_train_images(model):
    # Set the paths
    model_path = "data/model_weights.pt"
    image_dir = "data/train_images"
    output_dir = "data/cv_train_out"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {os.path.abspath(model_path)}")
        return

    # Check if the image directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {os.path.abspath(image_dir)}")
        return

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg'))]

    # Check if there are any image files
    if len(image_files) == 0:
        print(f"Error: No image files found in {os.path.abspath(image_dir)}")
        return

    print(f"Found {len(image_files)} image files in {os.path.abspath(image_dir)}")

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Read the image
        img = cv2.imread(image_path)
        
        # Check if the image was successfully loaded
        if img is None:
            print(f"Error: Failed to load the image at {os.path.abspath(image_path)}")
            continue
        
        print(f"Processing image {os.path.abspath(image_path)}")
        
        # Perform inference
        results = model(img)
        
        # Get the JSON output and parse it
        json_output = json.loads(results[0].tojson())
        
        # Get image name without extension
        image_name = os.path.splitext(image_file)[0]

        # Store the results for this image
        json_output_path = os.path.join(output_dir, f"cv_train_{image_name}.json")

        with open(json_output_path, "w") as json_file:
            json.dump(json_output, json_file, indent=2)

def main():
    # Set the paths
    model_path = "data/model_weights.pt"
    image_path = "data/train_images/p115.jpg"
    output_dir = "data/cv_train_out"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = YOLO(model_path)
    print("Model loaded successfully")

    predict_all_train_images(model)

    print(f"Processing complete. Check {output_dir} for the output files")
    
if __name__ == "__main__":
    main()