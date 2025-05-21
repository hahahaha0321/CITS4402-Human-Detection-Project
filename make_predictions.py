import numpy as np
import math
import cv2
from Feature_extraction import feature_extraction
from SVM_classifier import *
from sklearn import svm
import os
import pandas as pd
#import matplotlib.pyplot as plt

#Trains classifier, and tests it on image_folder, writes results into excel file
def make_predictions(training_folder = "Dataset/train", image_folder = "prediction_images", results_file = "predictions.xlsx"):
    
    classifier = train_SVM(training_folder)
    result = test_SVM(image_folder, classifier)
    predictions = result[7]
    image_names = result[8]
    print(predictions)
    print(image_names)
    
    # Create DataFrame
    df = pd.DataFrame({
        "filename": image_names,
        "prediction": predictions
    })

    # Save to Excel
    df.to_excel(results_file, index=False)
    print(f"Predictions saved to {results_file}")
    
    """
    prediction = 0
    human_image_path = os.path.join(image_folder, "human")
    nonhuman_image_path = os.path.join(image_folder, "non-human")

    
    human_images = [os.path.join(human_image_path, filename) for filename in os.listdir(human_image_path) if filename.endswith('.jpg') or filename.endswith('.png')]
    nonhuman_images = [os.path.join(nonhuman_image_path, filename) for filename in os.listdir(nonhuman_image_path) if filename.endswith('.jpg') or filename.endswith('.png')]

    detection_window_width = 18
    detection_window_height = 36
    
    x_step_size = detection_window_width//2
    y_step_size = detection_window_height//2
    for folder in [human_images,nonhuman_images]:
        for img_path in folder:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape[:2]
                             
            #slide detection window along image and check for human
            for y in range(0, img_height - detection_window_height + 1, y_step_size):
                for x in range(0, img_width - detection_window_width + 1, x_step_size):
                    detection_window = img[y:y + detection_window_height, x:x + detection_window_width]
                    hog_features = np.array(feature_extraction(detection_window)).flatten()
                    prediction_in_window = classifier.predict([hog_features])
                    
                    if prediction_in_window[0] == 1:
                        prediction = 1 #human was detected
                        break
            print(os.path.basename(img_path), "===", prediction)
        """
