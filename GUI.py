import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Global variables
model = None

# Function to load the saved model
def load_saved_model():
    global model
    model_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Model File", filetypes=[("H5 Model Files", "*.h5")])
    if model_path:
        model = load_model(model_path)
        update_status("Model loaded successfully!", "green")

# Function to classify uploaded image
def classify_image():
    global model
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((300, 300))  # Resize image for display
            img = ImageTk.PhotoImage(img)
            panel = tk.Label(root, image=img, bg="dark blue")
            panel.image = img  # Keep reference of the image
            panel.pack(pady=10)

            # Perform classification
            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_img_array = img_array / 255.0
            prediction = model.predict(normalized_img_array)
            if prediction[0][0] > 0.5:
                result = "Negative"
            else:
                result = "Positive"
            update_status(f"Prediction: {result}", "green")
        except Exception as e:
            update_status(f"Error: {str(e)}", "red")

# Function to update status message
def update_status(msg, color):
    status_label.config(text=msg, fg=color)

# GUI setup
root = tk.Tk()
root.title("COVID-19 Detection from X-ray Images")

# Configure dark blue color
bg_color = "dark blue"
fg_color = "white"

# Style the GUI
root.geometry("400x600")  # Set initial window size
root.config(bg=bg_color)  # Set background color

# Load Model Button
load_model_button = tk.Button(root, text="Load Model", font=("Helvetica", 14), bg=bg_color, fg=fg_color, command=load_saved_model)
load_model_button.pack(pady=20)

# Classify Image Button
classify_button = tk.Button(root, text="Classify Image", font=("Helvetica", 14), bg=bg_color, fg=fg_color, command=classify_image)
classify_button.pack(pady=20)

# Status Label
status_label = tk.Label(root, text="Status: Ready", font=("Helvetica", 12), bg=bg_color, fg="white")
status_label.pack(pady=20)

# Quit Button
quit_button = tk.Button(root, text="Quit", font=("Helvetica", 14), bg=bg_color, fg=fg_color, command=root.quit)
quit_button.pack(pady=20)

root.mainloop()
