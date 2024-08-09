import numpy as np
import pandas as pd
import cv2
import os


def load_data():
    # Load the training and test data
    # TODO: Majority of training annotations for images dont have complete annotations for all images. 
    # Use computer vision to infer the missing annotations?

    # Read in the training data and create a DataFrame
    training_data = []
    for filename in os.listdir("data/train_annotations"):
        if filename.endswith(".txt"):
            filepath = os.path.join("data/train_annotations", filename)
            with open(filepath, "r") as file:
                for line in file:
                    # Split the line into components: <object-class> <x> <y> <width> <height>
                    parts = line.strip().split()
                    object_class = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    # Append to the data list, including the id
                    training_data.append(
                        {
                            "pothole_number": int(os.path.splitext(filename)[0][1:]),
                            "object_class": object_class,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        }
                    )
    training_df = pd.DataFrame(training_data)
    training_labels = pd.read_csv("data/train_labels.csv")
    training_labels = training_labels.rename(
        columns={"Pothole number": "pothole_number"}
    )
    training_df = training_df.merge(training_labels, on="pothole_number", how="left")

    # Read in the test data
    test_labels = pd.read_csv("data/test_labels.csv")
    # TODO: Use computer vision to infer test annotations from test images?? Thoughts guys?
    return training_df, test_labels


def main():
    training_df, test_labels = load_data()
    print(training_df, test_labels)


if __name__ == "__main__":
    main()
