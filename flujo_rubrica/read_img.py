import numpy as np
import cv2
import pydicom
from PIL import Image

def read_dicom_file(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    pil_img = Image.fromarray(arr)
    arrf = arr.astype(float)
    m = arrf.max()
    arr8 = ((np.maximum(arrf, 0) / m) * 255.0).astype(np.uint8) if m > 0 else np.zeros_like(arrf, dtype=np.uint8)
    bgr = cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
    return bgr, pil_img  # (np BGR), (PIL)

def read_jpg_file(path):
    pil_img = Image.open(path).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, pil_img
