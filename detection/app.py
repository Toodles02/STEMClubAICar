import torch 
from model import predict
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog 


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

        image = Image.open(file_path).convert('RGB')

        prediction = predict(image)
        result_label.config(text=prediction)

root = tk.Tk()
root.title("Image Prediction Widget")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="None")
result_label.pack(pady=10)

root.mainloop()