import numpy as np
import cv2
import tensorflow as tf
from .preprocess_img import preprocess
from .load_model import model_fun

def _find_last_conv_layer(model, name_hint="conv10_thisone"):
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

    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
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
    return overlay_bgr[:, :, ::-1]

def grad_cam(array_bgr, class_idx=None, x_batch=None, alpha=0.8, colormap=cv2.COLORMAP_JET, conv_layer_name_hint="conv10_thisone"):
    model = model_fun()
    if x_batch is None:
        x_batch = preprocess(array_bgr)
    if class_idx is None:
        preds = model([x_batch], training=False).numpy()
        class_idx = int(np.argmax(preds if preds.ndim == 2 else preds[0]))
    conv_layer = _find_last_conv_layer(model, name_hint=conv_layer_name_hint)
    cam = _compute_cam(model, x_batch, class_idx=class_idx, conv_layer=conv_layer)
    overlay_rgb = _render_cam_on_image(cam, array_bgr, alpha=alpha, colormap=colormap)
    return overlay_rgb
