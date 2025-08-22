import numpy as np
import cv2

def preprocess(array_bgr):
    x = cv2.resize(array_bgr, (512, 512))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    x = clahe.apply(x)
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    return x.astype("float32")
