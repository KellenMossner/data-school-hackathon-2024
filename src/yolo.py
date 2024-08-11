from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime

# Set the paths
model_path = "data/model_weights.pt"
image_path = "data/train_images/p115.jpg"
output_dir = "data/cv_train_out"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create a log file for debug information
log_file_path = os.path.join(output_dir, "cv.log")
json_output_path = os.path.join(output_dir, "results.json")

# Function to write to log file
def write_to_log(message):
    with open(log_file_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")

# Check if the model file exists
if not os.path.exists(model_path):
    write_to_log(f"Error: Model file not found at {os.path.abspath(model_path)}")
    exit()

# Load the model
model = YOLO(model_path)
write_to_log("Model loaded successfully")

# Check if the image file exists
if not os.path.exists(image_path):
    write_to_log(f"Error: Image file not found at {os.path.abspath(image_path)}")
    exit()

# Read the image
img = cv2.imread(image_path)

# Check if the image was successfully loaded
if img is None:
    write_to_log(f"Error: Failed to load the image at {os.path.abspath(image_path)}")
    exit()

write_to_log("Image loaded successfully")

# Perform inference
results = model(img)
write_to_log("Inference completed")

# Get the JSON output
json_output = results[0].tojson()

# Write JSON output to file
with open(json_output_path, "w") as json_file:
    json.dump(json.loads(json_output), json_file, indent=2)

write_to_log(f"Results written to {json_output_path}")
print(f"Processing complete. Check {output_dir} for results and debug log.")