import tkinter as tk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET
import subprocess
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow import keras
import threading
import time  
from keras.preprocessing import image as keras_image 
from keras.models import load_model

current_image_path = None
image = None  # Imaginea curentă
result_text = "Zambet detectat:"
istoric_rezultate = []  # Inițializăm istoricul rezultatelor
camera_running = False
cap = None
empty_image = None
numar_captura = 1
root = tk.Tk()
emotion_model = load_model("best_model.h5")
buffer_size = 10

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_emotions(frame):
    global emotion_model, face_haar_cascade

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = keras_image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = emotion_model.predict(img_pixels)

        # Find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        cv2.putText(frame, f'Emotion: {predicted_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

    return frame
    


def detect_faces_and_smiles(image_path):
    global result_text, image, emotions, emotion_model
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        roi_gray = gray[y:y + h, x:x + w]
        
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        if len(smiles) > 0:
            accuracy = len(smiles) * 100 / len(faces)
            
            result_text = f'Smile Detected: {accuracy:.2f}%'
            cv2.putText(img, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    photo = ImageTk.PhotoImage(image)
    
    result_label.config(text=result_text)
    image_label.config(image=photo)
    image_label.image = photo

def import_action():
    global current_image_path, image
    file_path = filedialog.askopenfilename(filetypes=[("Imagini", "*.png *.jpg *.jpeg *.gif *.bmp")])
    if file_path:
        current_image_path = file_path
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)

        smile_detect = detect_smile(file_path)
        
        if smile_detect:
            result_text = "Zâmbet detectat: Da"
        else:
            result_text = "Zâmbet detectat: Nu"
        
        result_label.config(text=result_text)
        image_label.config(image=photo)
        image_label.image = photo
        with open("istoric_rezultate.txt", "a") as history_file:
            history_file.write(result_text + "\n")

def save_action():
    global current_image_path, image
    file_path = filedialog.askopenfilename(filetypes=[("Imagini", "*.png *.jpg *.jpeg *.gif *.bmp")])
    if file_path:
        current_image_path = file_path
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        
        detect_faces_and_smiles(file_path)

    if image is not None:
        ext = ".jpg"
        format_map = {
            "JPEG": (".jpeg", "*.jpeg"),
            "PNG": (".png", "*.png"),
            "GIF": (".gif", "*.gif"),
            "BMP": (".bmp", "*.bmp"),
            "TIFF": (".tiff", "*.tiff")
        }
        file_extension, file_type = format_map.get("JPEG", (".jpeg", "*.jpeg"))

        # Deschide fereastra de dialog de salvare cu numele implicit
        file_obj = filedialog.asksaveasfile(defaultextension=ext, filetypes=[("Imagini", f"*{file_extension}")])

        if file_obj:
            file_path = file_obj.name  
            
            image_rgb = image.convert("RGB")
            image_rgb.save(file_path, format="JPEG")
            print(f"Imaginea a fost salvată la: {file_path}")

def detect_smile(image_path):
    # Încărcați imaginea
    cv2.data.haarcascades
    smile_cascade_xml_url = cv2.data.haarcascades + 'haarcascade_smile.xml'
    smile_cascade = cv2.CascadeClassifier('C:\\Users\\Alex\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\cv2\\data\\haarcascade_smile.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

    if len(faces) > 0:
        return True
    else:
        return False
    
    return len(faces) > 0
    
def afiseaza_istoric():
    with open("istoric_rezultate.txt", "w") as f:
        for result in istoric_rezultate:
            image_path = result["image_path"]
            smile_detected = "Da" if result["smile_detected"] else "Nu"
            f.write(f"Imagine: {image_path}, Zâmbet detectat: {smile_detected}\n")

def show_history():
    subprocess.Popen(["notepad.exe", "istoric_rezultate.txt"])

def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)

def show_camera_feed():
    global cap, camera_running, image_label

    while camera_running:
        if cap is not None:
            ret, frame = cap.read()

            if not ret:
                break

            frame_with_emotions = analyze_emotions(frame)

            frame_with_emotions = cv2.cvtColor(frame_with_emotions, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)

            image_label.config(image=photo)
            image_label.image = photo

            root.update()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    release_camera()

def release_camera():
    global cap
    if cap is not None:
        cap.release()  # Eliberează resursele camerei
    cap = None
    image_label.config(image=empty_image)

def toggle_camera():
    global camera_running, cap

    if not camera_running:
        init_camera()  # Inițializează camera
        camera_running = True
        camera_button.config(text="Oprește Camera")
        show_camera_feed()
    else:
        camera_running = False
        camera_button.config(text="Pornire Camera")
        release_camera()
        image_label.config(image=None)

def captura_imagine():
    global cap, camera_running, numar_captura

    if camera_running and cap is not None:
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            image_label.config(image=photo)
            image_label.image = photo
            
            nume_fisier = f"captura_imaginii{numar_captura}.png"
            image.save(nume_fisier)
            print(f"Imaginea a fost salvată la: {nume_fisier}")
            
            numar_captura += 1

#def resize_all_images_in_directory(directory, new_size=(224, 224)):
#    for emotion_folder in os.listdir(directory):
#        for image_name in os.listdir(os.path.join(directory, emotion_folder)):
#            image_path = os.path.join(directory, emotion_folder, image_name)
            
#            if image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
#                img = Image.open(image_path)
#                img = img.resize(new_size)
#                img.save(image_path)

def deschide_fereastra_secundara():
    fereastra_secundara = tk.Toplevel(root)
    fereastra_secundara.title("Fereastră Secundară")
    fereastra_secundara.geometry("800x600")
    istoric_button = tk.Button(fereastra_secundara, text="Afișează Istoric", command=show_history)
    istoric_button.pack(side="top", fill="x", pady=10)
    performance_label = tk.Label(fereastra_secundara, text="Matrici de performanță:")
    performance_label.pack(side="top", fill="x", pady=5)

#analyze_emotions(emotion_model, face_haar_cascade)

root.title("Interfață pentru aplicație")
root.geometry("1200x800")

image_dir = "C:\\Users\\Alex\\Desktop\\1 py\\PythonApplication5\\PythonApplication5\\train"
new_size = (224, 224)
#resize_all_images_in_directory(image_dir, new_size)

left_frame = tk.Frame(root)
left_frame.pack(anchor="nw", padx=10, pady=10)

import_button = tk.Button(left_frame, text="Import", command=import_action)
import_button.pack(side="top", fill="x", pady=5)
image_label = tk.Label(left_frame)
image_label.pack(side="top", pady=5)

salvare = tk.Button(left_frame, text="Salvare", command=save_action)
salvare.pack(side="top", fill="x", pady=5)

result_label = tk.Label(left_frame, text=result_text)
result_label.pack(side="top", fill="x", pady=5)

camera_button = tk.Button(left_frame, text="Pornire Camera", command=toggle_camera)
camera_button.pack()

captura_button = tk.Button(left_frame, text="Capturare Imagine", command=captura_imagine)
captura_button.pack(side="top", fill="x", pady=5)

buton_deschide_fereastra = tk.Button(left_frame, text="Deschide Fereastra Secundară", command=deschide_fereastra_secundara)
buton_deschide_fereastra.pack(side="top", fill="x", pady=5)

root.configure(bg="lightgray") 
camera_button.configure(bg="green", fg="white") 
captura_button.configure(bg="blue", fg="white") 

empty_image = ImageTk.PhotoImage(Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)))
image_label = tk.Label(root)
image_label.pack()

menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu)
menu.add_cascade(label="Meniu", menu=file_menu)

root.mainloop()


