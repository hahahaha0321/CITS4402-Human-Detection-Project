import numpy as np
import math
import cv2
from Feature_extraction import feature_extraction
from sklearn import svm
import os
import random

#optimal parameters found in ablation study
num_bins = 133
block_size = 16
dx_kernel=np.array([[-1, 0, 1]])
dy_kernel=np.array([[1], [0], [-1]])
norm_technique = "L2-hys"
block_forming_pattern = "horizontal"

#trains SVM using default parameters for feature extraction
def train_SVM(training_folder =  r"dataset/train", num_bins=num_bins, block_size=block_size, dx_kernel=dx_kernel, dy_kernel=dy_kernel, norm_technique=norm_technique, block_forming_pattern=block_forming_pattern):
    
    human_image_path = training_folder + '/human'
    nonhuman_image_path = training_folder + '/non-human'
    print(f"Training SVM classifier - using images from \"{training_folder}\"")
    
    images = []
    y = []
    
    # Get the list of all human and non-human images
    human_images = [os.path.join(human_image_path, filename) for filename in os.listdir(human_image_path) if filename.endswith('.jpg') or filename.endswith('.png')]
    nonhuman_images = [os.path.join(nonhuman_image_path, filename) for filename in os.listdir(nonhuman_image_path) if filename.endswith('.jpg') or filename.endswith('.png')]
    
    print(f"Using {len(human_images)} human images and {len(nonhuman_images)} non-human images.")

    # Process human images
    for img_path in human_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            y.append(1)

    # Process non-human images
    for img_path in nonhuman_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            y.append(0)

    # Extract features using feature_extraction function
    X = [np.array(feature_extraction(img, num_bins, block_size, dx_kernel, dy_kernel, norm_technique = norm_technique, block_forming_pattern = block_forming_pattern)).flatten() for img in images]
    
    print("Starting SVM training...")
    classifier = svm.LinearSVC()
    classifier.fit(X, y)
    print("SVM training completed.")

    return classifier
    
def test_SVM(test_folder, classifier, num_bins=num_bins, block_size=block_size, dx_kernel=dx_kernel, dy_kernel=dy_kernel, norm_technique=norm_technique, block_forming_pattern=block_forming_pattern, verbose = False):
    print("\nTesting SVM Classifier - using images from  \"", test_folder, "\"")
    human_image_path = test_folder + '/human'
    nonhuman_image_path = test_folder + '/non-human'
    images = []
    y = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    #detection window this to match the size of testing data
    detection_window_width = 64 #18
    detection_window_height = 128 #36
    
    x_step_size = detection_window_width//2
    y_step_size = detection_window_height//2
        
    # Get the list of all human and non-human images
    human_images = [os.path.join(human_image_path, filename) for filename in os.listdir(human_image_path) if filename.endswith('.jpg') or filename.endswith('.png')]
    nonhuman_images = [os.path.join(nonhuman_image_path, filename) for filename in os.listdir(nonhuman_image_path) if filename.endswith('.jpg') or filename.endswith('.png')]
    print(f"Using {len(human_images)} human images and {len(nonhuman_images)} non-human images.")
    
    
    
    predictions = []
    file_names = []
    
    for img_path in human_images:
        file_names.append(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            #slides detection window across larger image and returns 1 as soon as human is detected using feature extraction on detection window
            prediction = 0
            img_height, img_width = img.shape[:2]
            for y in range(0, img_height - detection_window_height + 1, y_step_size):
                for x in range(0, img_width - detection_window_width + 1, x_step_size):
                    detection_window = img[y:y + detection_window_height, x:x + detection_window_width]
                    hog_features = np.array(feature_extraction(detection_window, num_bins, block_size, dx_kernel, dy_kernel, norm_technique = norm_technique, block_forming_pattern = block_forming_pattern)).flatten()
                    prediction_in_window = classifier.predict([hog_features])
                    
                    if prediction_in_window[0] == 1:
                        prediction = 1 #human was detected
                        break
                    
            #print("Prediction for ", os.path.basename(img_path), " is ", prediction, " expected ", 1)
            predictions.append(prediction)
            if prediction == 1:
                TP += 1
            else:
                FN += 1
                
    for img_path in nonhuman_images:
        file_names.append(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            prediction = 0
            img_height, img_width = img.shape[:2]
            for y in range(0, img_height - detection_window_height + 1, y_step_size):
                for x in range(0, img_width - detection_window_width + 1, x_step_size):
                    detection_window = img[y:y + detection_window_height, x:x + detection_window_width]

                    hog_features = np.array(feature_extraction(detection_window, num_bins, block_size, dx_kernel, dy_kernel, norm_technique = norm_technique, block_forming_pattern = block_forming_pattern)).flatten()
                    prediction_in_window = classifier.predict([hog_features])
                    if prediction_in_window[0] == 1:
                        prediction = 1 #human was detected
                        break
                    
            predictions.append(prediction)
            #print("Prediction for ", os.path.basename(img_path), " is ", prediction, " expected ", 0)
            if prediction == 0:
                TN += 1
            else:
                FP += 1
    
    accuracy = 0
    precision = 0
    recall = 0
    if (TP+TN+FP+FN != 0):
        accuracy = (TP+TN)/(TP+TN+FP+FN)    
    if (TP+FP != 0):
        precision =  TP/(TP+FP)
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    
    if (verbose):
        print("RESULTS:")
        print("True pos= ", TP, "False pos= ", FP)
        print("True neg= ", TN, "False Neg= ", FN)
        print("Accuracy=", accuracy)
        print("precision=", precision)
        print("Recall=", recall)
        print("Miss rate", FN / (TP + FN))
        
    
    return (TP, FP, TN, FN, accuracy, precision, recall, predictions, file_names)

#trains and tests SVM classifier 
def main(datasetFolder = "dataset", verbose = False):
    
    print("Starting training...")
    trainingFolder = datasetFolder + "/train"
    testingFolder = datasetFolder + "/test"
    
    classifier = train_SVM(trainingFolder)
    classifier1 = train_SVM(trainingFolder, norm_technique = "L1")
    print("Training completed. Starting testing...")
    test_SVM(testingFolder, classifier, verbose=verbose)
    test_SVM(testingFolder, classifier1, verbose=verbose)
