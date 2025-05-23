import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import shutil
import time
import numpy as np
from feature_extraction import feature_extraction

from joblib import load #imports the SVM classifier that has been trained in train_SVM
clf = load('svm_model.joblib')

# === GUI Class ===
class HumanDetectorGUI:
    def __init__(self, master):
            # Top frame for image
        top_frame = tk.Frame(master)
        top_frame.pack(side=tk.TOP, pady=10)
        
        # Title label goes at the top
        self.title_label = tk.Label(top_frame, text="Human Detection", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=(0, 10))

        # Then the canvas below it
        self.image_canvas = tk.Canvas(top_frame, width=200, height=400, bg="white", highlightthickness=1, highlightbackground="black")
        self.image_canvas.pack()
        
        left_frame = tk.Frame(top_frame)
        left_frame.pack(side=tk.LEFT, padx=10, anchor='n')

        self.image_name_label = tk.Label(left_frame, text="No image", font=("Arial", 14))
        self.image_name_label.pack(anchor='nw')

        self.prediction_label = tk.Label(left_frame, text="predict", font=("Arial", 14))
        self.prediction_label.pack(anchor='nw', pady=(5,0))

        
        # Middle frame for buttons
        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)

        self.load_dir_button = tk.Button(button_frame, text="Load Directory", command=self.load_dir)
        self.load_dir_button.pack(side=tk.LEFT, padx=10)

        self.change_images_button = tk.Button(button_frame, text="Change Images", command=self.change_images)
        self.change_images_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = tk.Button(button_frame, text="Show Predictions", command=self.show_predictions, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=10)



        self.result_label = tk.Label(master, text="", font=("Arial", 14))
        self.result_label.pack()


        self.master = master
        master.title("Human Detection Viewer")

        self.image_index = 0
        self.images = []
        
        # Create a label to display the chosen image
        self.image_label = tk.Label(master, width=400, height=400, bg="white")
        self.image_label.pack(side=tk.TOP, padx=5, pady=5)
       
        self.filename_label = tk.Label(master, text="", font=("Arial", 12))
        self.filename_label.pack()


        self.label = tk.Label(master, text="Load a folder with images:")
        self.label.pack()
        
       
        self.result_label = tk.Label(master, text="", font=("Arial", 14))
        self.result_label.pack()
    
    
    #opens dialog box to select what images to use for predictions
    def load_dir(self):
        predictions_dir = filedialog.askdirectory(title="Select folder with images", initialdir =os.path.dirname(os.path.abspath(__file__)) )
        self.images = []
        
        human_dir = os.path.join(predictions_dir, r"human")
        nonhuman_dir = os.path.join(predictions_dir, r"non-human")
        
        for folder in [human_dir, nonhuman_dir]:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Just mark them as "Unknown" since no label info here
                    self.images.append(fpath)
                    
        self.predict_button.config(state=tk.NORMAL)

        
    #Randomly copies 10 human and 10 non-human images from dataset directory into the predictions folder
    def change_images(self):
        target_dir= r"Dataset\test"
        dest_dir = r"prediction_images"
        target_human_dir = os.path.join(target_dir, r"human")
        target_nonhuman_dir = os.path.join(target_dir, r"non-human")
        dest_human_dir = os.path.join(dest_dir, r"human")
        dest_nonhuman_dir = os.path.join(dest_dir, r"non-human")
        
        #Clear predicitions folder 
        for folder in [dest_human_dir, dest_nonhuman_dir]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    
       # Randomly sample 10 images from each folder
        human_images = [f for f in os.listdir(target_human_dir) if os.path.isfile(os.path.join(target_human_dir, f))]
        nonhuman_images = [f for f in os.listdir(target_nonhuman_dir) if os.path.isfile(os.path.join(target_nonhuman_dir, f))]
        sampled_human_images = random.sample(human_images, 10)
        sampled_nonhuman_images = random.sample(nonhuman_images, 10)
        
         
        # Copy sampled images into prediction folder
        for image in sampled_human_images:
            shutil.copy(os.path.join(target_human_dir, image), os.path.join(dest_human_dir, image))

        for image in sampled_nonhuman_images:
            shutil.copy(os.path.join(target_nonhuman_dir, image), os.path.join(dest_nonhuman_dir, image))

    
    #iterates through each image showing prediction, then shows all images on one screen with predictions
    def show_predictions(self):
        if not self.images:
            return

        if self.image_index >= len(self.images):
            self.image_index = 0  # Reset to start or remove this line to stop
        
        
                    
                    
        img_path = self.images[self.image_index]
        img = Image.open(img_path)
        
        hog_features = np.array(feature_extraction(img)).flatten()
        prediction = clf.predict([hog_features])
        
        img = img.resize((200, 400))  # Resize as needed
        
        self.photo = ImageTk.PhotoImage(img)
        self.image_canvas.delete("all")  # Clear previous image
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.filename_label.config(text=os.path.basename(img_path))
        
        self.image_name_label.config(text=os.path.basename(img_path))
        if (prediction == 0):
            self.prediction_label.config(text="NON-HUMAN")
        else:
            self.prediction_label.config(text="NON-HUMAN");
        
        self.image_index += 1
        self.master.after(2000, self.show_predictions)
    
            

# === Run the GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    gui = HumanDetectorGUI(root)
    root.mainloop()
