#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog

from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import csv
import tkcap
import numpy as np
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.experimental.output_all_intermediates(True)
import cv2
import os
import pydicom 
from flujo_rubrica.read_img import read_dicom_file, read_jpg_file
from flujo_rubrica.integrator import predict

_MODEL = None

def model_fun():
   
    global _MODEL
    if _MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), "modelo", "conv_MLP_84.h5") 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
        _MODEL = tf.keras.models.load_model(model_path, compile=False)
        try:
            print("Modelo cargado OK:", _MODEL.name, "| input_shape:", getattr(_MODEL, "input_shape", None))
        except Exception:
            pass
    return _MODEL

def _find_last_conv_layer(model, name_hint="conv10_thisone"):
    """Devuelve una capa convolucional válida para Grad-CAM (por nombre o por tipo)."""
    if name_hint:
        try:
            return model.get_layer(name_hint)
        except (ValueError, KeyError):
            pass
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not convs:
        raise ValueError("No se encontró capa convolucional para Grad-CAM.")
    return convs[-1]

def _compute_cam(model, x_batch, class_idx=None, conv_layer=None):
   
    if conv_layer is None:
        conv_layer = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(model.inputs, [conv_layer.output, model.output])

    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32) #Agrego esta linea agregada para quitar el warning de keras
# Con esto se logra estandarizar la esrada en tensor con la salida que estpy generando.

    with tf.GradientTape() as tape:
        #conv_out, preds = grad_model(x_batch, training=False) 
        conv_out, preds = grad_model([x_batch], training=False)  
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)   
    conv_out = conv_out[0]                  
    grads = grads[0]                        
    weights = tf.reduce_mean(grads, axis=(0, 1))                         
    cam = tf.reduce_sum(conv_out * tf.cast(weights, conv_out.dtype), axis=-1)  

    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

def _render_cam_on_image(cam, base_bgr, alpha=0.8, colormap=cv2.COLORMAP_JET):
  
    base = base_bgr
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.shape[2] == 1:
        base = cv2.cvtColor(base[:, :, 0], cv2.COLOR_GRAY2BGR)

    H, W = base.shape[:2]
    heat = cv2.resize(cam, (W, H))
    heat_u8 = (255.0 * np.clip(heat, 0.0, 1.0)).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat_u8, colormap)
    overlay_bgr = cv2.addWeighted(heatmap, alpha, base, 1.0 - alpha, 0.0)
    return overlay_bgr[:, :, ::-1]  # BGR -> RGB

def grad_cam(array, class_idx=None, x_batch=None, alpha=0.8, colormap=cv2.COLORMAP_JET, conv_layer_name_hint="conv10_thisone"):
  
    model = model_fun()

    # Pepara un bash sólo si no lo recibimos ya
    if x_batch is None:
        x_batch = preprocess(array).astype("float32")
    else:
        x_batch = x_batch.astype("float32")

    # Si no se da class_idx, se decide aquí 
    if class_idx is None:
        preds = model([x_batch], training=False).numpy()# Se le puso corchetes para evitar error de keras
        #Asi continuamos con la matriz en x_batch 
        class_idx = int(np.argmax(preds if preds.ndim == 2 else preds[0]))

    conv_layer = _find_last_conv_layer(model, name_hint=conv_layer_name_hint)
    cam = _compute_cam(model, tf.convert_to_tensor(x_batch), class_idx=class_idx, conv_layer=conv_layer)
    overlay_rgb = _render_cam_on_image(cam, array, alpha=alpha, colormap=colormap)
    return overlay_rgb
  


def predict(array):
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = preprocess(array).astype("float32")
    #   2. call function to load model and predict: it returns predicted class and probability
    model = model_fun()
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    #preds = model(batch_array_img, training=False).numpy()
    preds = model([batch_array_img], training=False).numpy()
    prediction = int(np.argmax(preds))
    proba = float(np.max(preds)) * 100.0
    #prediction = np.argmax(model.predict(batch_array_img))
    #proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    heatmap = grad_cam(array)
    return (label, proba, heatmap)


def read_dicom_file(path):
    img = pydicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
  
    try:
        pil_img = Image.open(path).convert("RGB")   
    except Exception as e:
        raise ValueError(f"No se pudo abrir la imagen con PIL: {path}\n{e}")

    rgb = np.array(pil_img)                         
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)     
    return bgr, pil_img


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg;*.jpg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
                ("todos","*.*")
            ),
        )
        if not filepath:
            return
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".dcm":
            self.array, img2show = read_dicom_file(filepath)
        else: 
             self.array, img2show = read_jpg_file(filepath)
        RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
        self.img1 = img2show.resize((250, 250), RESAMPLE)
        self.img1 = ImageTk.PhotoImage(self.img1)
        self.text_img1.image_create(END, image=self.img1)
        self.button1["state"] = "enabled"

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
        self.img2 = self.img2.resize((250, 250), RESAMPLE)
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(self.img1, "end")
            self.text_img2.delete(self.img2, "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
