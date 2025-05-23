#Used to run ablation studies and visualise the results

import numpy as np
import math
import cv2
from Feature_extraction import feature_extraction
from SVM_classifier import *
from sklearn import svm
import os
import matplotlib.pyplot as plt

class test_result():
    def __init__(self, classifier, num_bins, block_size, dx_kernel, dy_kernel, TP=None, FP=None, TN=None, FN=None, accuracy=None, precision=None, recall=None):
        self.classifier = classifier
        self.num_bins = num_bins
        self.block_size = block_size
        self.dx_kernel = dx_kernel
        self.dy_kernel = dy_kernel #return (TP, FP, TN, FN, accuracy, precison, recall)
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall


#Performs ablation study of number of bins and plots result
def test_num_bins(datasetDir = r"dataset"):
    training_folder = datasetDir + r"/train"
    testing_folder = datasetDir + r"/test"
    
    num_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180] #number of bins to be tested  
    #num_bins = list(range(125,136,1))
    TP = []
    FP = []
    TN = []
    FN = []
    accuracy = []
    precision = []
    recall = []
    test_num = 0
    num_tests = len(num_bins)
    for i in num_bins:
        classifier = train_SVM(training_folder, num_bins=i)
        result = test_SVM(testing_folder, classifier, num_bins =  i)
        
        #result = test_result(classifier, num_bins, block_size, dx_kernel, dy_kernel) #(TP, FN, TN, FN, accuracy, precision, recall)
        TP.append(result[0])
        FP.append(result[1])
        TN.append(result[2])
        FN.append(result[3])
        accuracy.append(result[4])
        precision.append(result[5])
        recall.append(result[6])
            
        test_num += 1
        print("\033[32m\n", test_num, "/", num_tests, "complete\n\033[0m")
    
    #print table of results
    headers = [ "Num_bins", "TP", "FP", "TN", "FN",  "Accuracy", "Precision", "Recall"]
    template = ""
    for i in range(len(headers)):
        template += "|{" + str(i) + ":<" + str(len(headers[i]) + 5) + "}"
          
    print(template.format(*headers))  

    for i in range(len(num_bins)):
        print(template.format(f'"{num_bins[i]}"', TP[i], FP[i], TN[i], FN[i], f"{accuracy[i]:.3f}", f"{precision[i]:.3f}", f"{recall[i]:.2f}"))
    
    #visualise results
    plt.figure(figsize=(10, 6))
    plt.plot(num_bins, accuracy, label='Accuracy', marker='o')
    plt.plot(num_bins, precision, label='Precision', marker='s')
    plt.plot(num_bins, recall, label='Recall', marker='^')
    
    plt.title('Performance Metrics vs. Number of Bins')
    plt.xlabel('Number of Bins')
    plt.ylabel('Metric Value')
    plt.xticks(num_bins)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    
    
 #Performs ablation study on block size and plots result
def test_block_size(datasetDir = r"dataset"):
    training_folder = datasetDir + r"/train"
    testing_folder = datasetDir + r"/test"
    
    block_sizes = [8,16, 24, 32]
    TP = []
    FP = []
    TN = []
    FN = []
    accuracy = []
    precision = []
    recall = []
    test_num = 0
    num_tests = len(block_sizes)
    for i in block_sizes:
        classifier = train_SVM(training_folder, block_size=i)
        result = test_SVM(testing_folder, classifier, block_size =  i)
        
        TP.append(result[0])
        FP.append(result[1])
        TN.append(result[2])
        FN.append(result[3])
        accuracy.append(result[4])
        precision.append(result[5])
        recall.append(result[6])
            
        test_num += 1
        print("\033[32m\n", test_num, "/", num_tests, "complete\n\033[0m")
    
    #prints table
    headers = [ "Block Size", "TP", "FP", "TN", "FN",  "Accuracy", "Precision", "Recall"]
    template = ""
    for i in range(len(headers)):
        template += "|{" + str(i) + ":<" + str(len(headers[i]) + 5) + "}"
          
    print(template.format(*headers))  #

    for i in range(len(block_sizes)):
        print(template.format(f'"{block_sizes[i]}"', TP[i], FP[i], TN[i], FN[i], f"{accuracy[i]:.3f}", f"{precision[i]:.3f}", f"{recall[i]:.2f}"))

    #visualise results
    plt.figure(figsize=(10, 6))
    plt.plot(block_sizes, accuracy, label='Accuracy', marker='o')
    plt.plot(block_sizes, precision, label='Precision', marker='s')
    plt.plot(block_sizes, recall, label='Recall', marker='^')


    plt.title('Performance Metrics vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Metric Value')
    plt.xticks(block_sizes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
       
    
    
 #Performs ablation study of normalisation technique (both normalisation and block forming pattern) and plots result
def test_norm_technique(datasetDir = r"dataset"):
    training_folder = datasetDir + r"/train"
    testing_folder = datasetDir + r"/test"
    
    norm_techniques = ["L2-hys", "None", "L1", "L2", "L2-hys", "soft-max"]
    block_forming_patterns = ["horizontal","TL_square", "plus", "X", "neighbours", "vertical", "horizontal"]
    results = {norm: [] for norm in norm_techniques} #dictionary storing results in the form [norm_technique1: {[block_forming_pattern1, result], [block_forming_patterns2, result]
    test_num = 0
    num_tests = len(norm_techniques) *len(block_forming_patterns)
    for block_pattern in block_forming_patterns:
        for norm in norm_techniques:
            classifier = train_SVM(training_folder, norm_technique=norm, block_forming_pattern=block_pattern)
            result = test_SVM(testing_folder, classifier, norm_technique =  norm, block_forming_pattern=block_pattern)
            results[norm].append([])
            results[norm][-1] = [block_pattern]
            results[norm][-1].append(result)
            test_num += 1
            print("\033[32m\n", test_num, "/", num_tests, "complete\n\033[0m")
    
    
    #prints table
    headers = [ "Normalisation Technique", "Block pattern", "TP", "FP", "TN", "FN",  "Accuracy", "Precision", "Recall"]
    template = ""
    for i in range(len(headers)):
        template += "|{" + str(i) + ":<" + str(len(headers[i]) + 5) + "}"
          
    print(template.format(*headers))      
    
    for r in results:
        for i in range(len(results[r])):
            print(template.format(f'"{r}"', results[r][i][0], results[r][i][1][0],  results[r][i][1][1],  results[r][i][1][2], results[r][i][1][3], f"{results[r][i][1][4]:.3f}", f"{results[r][i][1][5]:.3f}", f"{results[r][i][1][6]:.3f}"))

    #visualise results
    for norm_technique in norm_techniques:
        block_patterns = [entry[0] for entry in results[norm_technique]]
        accuracies = [entry[1][4] for entry in results[norm_technique]]
        plt.plot(block_patterns, accuracies, label=f'{norm_technique}', marker='o')

    plt.title('Accuracy vs. Block Forming Pattern for Each Normalisation Technique')
    plt.xlabel('Block Forming Pattern')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
#ablation study on gradient filters
def test_gradient_filters(datasetDir = r"dataset"):
    training_folder = datasetDir + r"/train"
    testing_folder = datasetDir + r"/test"
    
    gradient_filters = ["[-1,0,1]", "sobel", "prewitt", "scharr"]
    
    # X-direction kernels for each filter
    dx_kernels = [
        np.array([[-1, 0, 1]]),  # Simple gradient
        np.array([[-1, 0, 1],    [-2, 0, 2], [-1, 0, 1]]), # sobel
        np.array([[-1, 0, 1],    [-1, 0, 1], [-1, 0, 1]]), # prewitt
        np.array([[-3, 0, 3],    [-10, 0, 10], [-3, 0, 3]]), # scharr
    ]

    # Y-direction kernels for each filter
    dy_kernels = [
        np.array([[1], [0], [-1]]),  # Simple gradient
        np.array([[1,  2,  1],  [0,  0,  0], [-1, -2, -1]]), #sobel
        np.array([[1,  1,  1],  [0,  0,  0], [-1, -1, -1]]),#prewitt
        np.array([[3,10,3],  [0,  0,  0], [-3, -10, -3]])#scharr
        
        ]
         
    TP = []
    FP = []
    TN = []
    FN = []
    accuracy = []
    precision = []
    recall = []
    test_num = 0
    num_tests = len(dx_kernels)
    for i in range(num_tests):
        classifier = train_SVM(training_folder, dx_kernel = dx_kernels[i], dy_kernel = dy_kernels[i])
        result = test_SVM(testing_folder, classifier, dx_kernel = dx_kernels[i], dy_kernel = dy_kernels[i])
    
        TP.append(result[0])
        FP.append(result[1])
        TN.append(result[2])
        FN.append(result[3])
        accuracy.append(result[4])
        precision.append(result[5])
        recall.append(result[6])
            
        test_num += 1
        print("\033[32m\n", test_num, "/", num_tests, "complete\n\033[0m")
    
    #print table
    headers = [ "Gradient filterl", "TP", "FP", "TN", "FN",  "Accuracy", "Precision", "Recall"]
    template = ""
    for i in range(len(headers)):
        template += "|{" + str(i) + ":<" + str(len(headers[i]) + 5) + "}"
          
    print(template.format(*headers))  # Pass elements of header as arguments to format

    #Visualise results
    for i in range(len(dx_kernels)):
        print(template.format(f'"{gradient_filters[i]}"',TP[i], FP[i], TN[i], FN[i], f"{accuracy[i]:.3f}", f"{precision[i]:.3f}", f"{recall[i]:.2f}"))

    plt.figure(figsize=(10, 6))
    plt.plot(gradient_filters, accuracy, label='Accuracy', marker='o')
    plt.plot(gradient_filters, precision, label='Precision', marker='s')
    plt.plot(gradient_filters, recall, label='Recall', marker='^')

    plt.title('Performance Metrics vs. Gradient Filter')
    plt.xlabel('Gradient Filter')
    plt.ylabel('Metric Value')
    plt.xticks(gradient_filters)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
       


