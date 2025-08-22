import numpy as np
from .preprocess_img import preprocess
from .load_model import model_fun
from .grad_cam import grad_cam

LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}

def predict(array_bgr):
    x = preprocess(array_bgr)  # (1,512,512,1) float32
    model = model_fun()
    preds = model([x], training=False).numpy()
    idx = int(np.argmax(preds))
    proba = float(np.max(preds)) * 100.0
    label = LABELS.get(idx, str(idx))
    heatmap = grad_cam(array_bgr, class_idx=idx, x_batch=x)
    return label, proba, heatmap
