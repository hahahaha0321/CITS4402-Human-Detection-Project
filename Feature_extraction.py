import numpy as np
import math
import cv2


#normalization scheme used in research paper
def l2_hys_normalize(v):
    v = np.array(v, dtype=np.float32)  #Change to numpy array for element wise operations
    epsilon = 1e-5
    norm = np.sqrt(np.sum(v ** 2) + epsilon ** 2)
    v = v / norm
    
    #limit max value
    v = np.clip(v, 0, 0.2)
    
    #normalise again
    norm = np.sqrt(np.sum(v ** 2) + epsilon ** 2)
    v = v / norm
    
    v= v.tolist() #convert back to list
    return v

#normalization scheme used in research paper but without clipping
def l2_normalize(v):
    v = np.array(v, dtype=np.float32)  #Change to numpy array for element wise operations
    epsilon = 1e-5
    norm = np.sqrt(np.sum(v ** 2) + epsilon ** 2)
    v = v / norm
    
    v= v.tolist() #convert back to list
    return v
  
#Normalisation - scales length of vector to 1
def l1_normalize(v):
    v = np.array(v, dtype=np.float32)
    norm = np.sum(np.abs(v))  # sum of absolute values
    v = v / norm
    return v.tolist()

def softmax_normalize(v):
    v = np.array(v, dtype=np.float32)
    exps = np.exp(v - np.max(v))
    v = exps / np.sum(exps)
    return v.tolist()

    
def feature_extraction(image, num_bins = 133,  block_size = 16, dx_kernel = np.array([[-1, 0, 1]]), dy_kernel = np.array([[1], [0], [-1]]), detection_window_width = 64, detection_window_height = 128,  norm_technique = "L2-hys", block_forming_pattern = "TL_square"):
    #print(norm_technique)
    #Resize image
    detection_window_width = 64
    detection_window_height = 128
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale if needed
    image = cv2.resize(image, (detection_window_width, detection_window_height))   
    
    #Calculating derivatives
    Gx = cv2.filter2D(image, cv2.CV_64F, dx_kernel) #x derivatives of image 
    Gy = cv2.filter2D(image, cv2.CV_64F, dy_kernel) #y derivatives of image
            
     #Derivative orientations and magnitudes
    gradient_mag = (Gx**2 + Gy**2)**0.5
    gradient_orientation = np.arctan2(Gy, Gx)  #in radians
    #print("Oreintation max and min: ", np.min(gradient_orientation), " and ", np.max(gradient_orientation))
    gradient_orientation*=(180.0/math.pi) #convert to degrees
    gradient_orientation%=180.0 #normalise to 0-180 degree range
    
    #print("Gradinet max and min: ", np.min(gradient_mag), " and ", np.max(gradient_mag))
    #print("Oreintation max and min: ", np.min(gradient_orientation), " and ", np.max(gradient_orientation))
    #Split image into cells
    cell_size = 8 
    image_histograms = [[[] for _ in range(detection_window_width // cell_size)] for _ in range(detection_window_height // cell_size)] #stores histogram of each cell which has dimensions num_binsx1
    for y in range(0, detection_window_height, cell_size):
        for x in range(0, detection_window_width, cell_size):
            cell_mag = gradient_mag[y:y+8, x:x+8]  #stores gradient magnitudes of cell
            cell_dir = gradient_orientation[y:y+8, x:x+8]  #stores gradient orientations of cells
           

            #Making histogram
            hist = [0] * num_bins
            bin_size = 180/num_bins
            for row in cell_dir:
                for col in row:
                    hist[int(col//bin_size)]+=1
            image_histograms[y//cell_size][x//cell_size] = hist
    
    #create blocks by combining cells
    if block_forming_pattern == "TL_square":
        block_pattern = [(0,0), (1,0), (0,1), (1,1)] #makes block from cell to left, down, down-left
    elif block_forming_pattern == "plus":
        block_pattern = [(0,0), (1,0), (-1,0), (0,1), (0,-1)] #makes block from cells in "+" config
    elif block_forming_pattern == "X":
        block_pattern = [(0,0), (1,1), (-1,-1), (-1,1), (1,-1)] #makes block from cells in "X" config i.e. diagonals
    elif block_forming_pattern == "neighbours":
        block_pattern =[(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (-1,1), (1,-1)] #makes block from all adjacent cells
    elif block_forming_pattern == "vertical":
        block_pattern = [(0,0), (0,1), (0,2)] #makes block from 2 cells below
    elif block_forming_pattern == "horizontal":
        block_pattern = [(0,0), (1,0), (2,0)] #makes block from 2 cells below
        
    """  
        
    print("Block forming")
    print(block_forming_pattern)
    print(" and so ")
    print(block_pattern)
    """
    blocks_histograms = []
    for y in range(len(image_histograms)):
        for x in range(len(image_histograms[0])):
            block_hist = []
            for coord in block_pattern:
                if y+coord[1] >= len(image_histograms) or x+coord[0] >= len(image_histograms[0]):
                    break
                block_hist += image_histograms[y+coord[1]][x+coord[0]]
                
            #normalise block
            if (norm_technique == "L1"):
                block_hist = l1_normalize(block_hist)
            elif (norm_technique == "L2"):
                block_hist = l2_normalize(block_hist)
            elif (norm_technique == "L2-hys"):
                block_hist = l2_hys_normalize(block_hist)
            elif (softmax_normalize == "soft-max"):
                block_hist = l2_hys_normalize(block_hist)
            elif (norm_technique == "None"):
                pass
            blocks_histograms += block_hist
                
    #normalising in 16x16 blocks
    #creates blocks by combining target cell with cell on right, cell below, cell diagonal right below
    """
    blocks_histograms = [[[] for _ in range(detection_window_width//cell_size + 1 - block_size//cell_size)] for _ in range(detection_window_height//cell_size +1 - block_size//cell_size)]
    for y in range(len(blocks_histograms)):
        for x in range(len(blocks_histograms[y])):
            hist = []
            for i in range(block_size//cell_size):
                for j in range(block_size//cell_size):
                    hist += image_histograms[y+i][x+j]
    """          
            
    #blocks_histograms[y][x] = hist
    
    
    #num_features = len(blocks_histograms) * len(blocks_histograms[0]) * len(blocks_histograms[0][0])
    #print("Number of features =", len(blocks_histograms))
    
    return blocks_histograms


    
    
