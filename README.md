# Patch Perfect: Using AI to Fix Roads
![Untitled design](https://github.com/user-attachments/assets/49062826-bdb8-45a3-9d5d-3c6502d254f4)
Can we predict the amount of asphalt required to fill a pothole given just an image of that pothole?
By using technologies such as computer vision and machine learning techniques we can estimate the number of bags of asphalt required to fill a pothole. 

## Computer Vision

We utilized the YOLOv8 large segmentation model to detect potholes. YOLOv8 is a state-of-the-art computer vision model known for its accuracy and efficiency in object detection tasks. To train the model, we leveraged the power of the Google Cloud Platform (GCP) and trained it on a large dataset of pothole images. Through iterations of training and fine-tuning over 200 epochs, we optimized the model's performance to achieve reliable pothole detection.

## Model Training
TODO

## Execution
1. Run the Computer Vision model to segment images.
```bash
python3 src/segment.py
```

2. Run the script to generate the final output.
```bash
python3 src/predict.py
```

### Team: Chi-Squared and Confused:
| Name           | SU Number |
|----------------|-----------|
| David Nicolay  | 26296918  |
| Jonty Donald   | 25957848  |
| Justin Dietrich| 25924958  |
| James Milne    | 25917307  |
| Kellen Mossner | 26024284  |S
