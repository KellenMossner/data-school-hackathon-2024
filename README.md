# Patch Perfect: Using AI to Fix Roads
![Untitled design](https://github.com/user-attachments/assets/49062826-bdb8-45a3-9d5d-3c6502d254f4)
Can we predict the amount of asphalt required to fill a pothole given just an image of that pothole?
By using technologies such as computer vision and machine learning techniques we can estimate the number of bags of asphalt required to fill a pothole. 

## Computer Vision
Making use of Roboflow, we manually labeled all the training images into the 3 classes, using polygons. We made use of a deep learning virtual machine on Google Cloud Platform (GCP) to train the YOLOv8 image segmentation model which we SSHed into. It took about 2.5 hours on a Nvidia Tesla T4 GPU with 200 epochs. We had trouble exporting the results to a JSON format for our prediction part since the mask which the model outputted was obviously separated by the stick, therefore the built in YOLOv8 JSON converted only gave us half the pothole. Using Convex Hulls we created our own output function to push the segmentation predictions for each image to a JSON format which included an outline of the entire pothole, even over the stick.

## Prediction
Initially a linear model was used to gather insight into which predictors provided primary value to the prediction power. Area, being central to predictions, was scaled using the known L1 length (estimated that the size corner to corner was 503.5 since our segmentation produced a polygon). We then went about an extensive feature engineering process with noticeable improvements from: Aspect Ratio, Max Diameter, L2 Length, and adding Pothole/L1/L2 confidence scores from the YOLOv8 prediction pushed our R-squared up. After exploring a wide variety of models we settled on the GradientBoostingRegressor from scikit-learn. After including all interaction terms (93 features total), the 10-fold Cross-Validation R-squared was about 0.5, however this yielded worse results on the Kaggle test set probably due to overfitting. We tried many external changes to the pipeline such as removing duplicate pothole classifications.

## Execution
NOTE: python 3.10 is required for the scripts to run.
1. Run the Computer Vision model to segment images.
```bash
python3 src/segment.py
```

2. Run the script to generate the final output.
```bash
python3 src/predict.py
```

## Using the API
![image](https://github.com/user-attachments/assets/42f2c22e-ba13-4a8d-8247-3de6ba50ed5e)
1. Start the API script.
```bash
python3 src/app.py
```

2.  Go to the link printed in the terminal and follow the instructions. If no link is printed go to http://localhost:5000/.

### Team: Chi-Squared and Confused:
| Name           | SU Number |
|----------------|-----------|
| David Nicolay  | 26296918  |
| Jonty Donald   | 25957848  |
| Justin Dietrich| 25924958  |
| James Milne    | 25917307  |
| Kellen Mossner | 26024284  |S
