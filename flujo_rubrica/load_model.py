import os
import tensorflow as tf

_MODEL = None

def model_fun(model_filename: str = "conv_MLP_84.h5"):
    """Carga el modelo como singleton desde ./modelo/<model_filename>."""
    global _MODEL
    if _MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "modelo", model_filename)
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
        _MODEL = tf.keras.models.load_model(model_path, compile=False)
        try:
            print("Modelo cargado OK:", _MODEL.name, "| input_shape:", getattr(_MODEL, "input_shape", None))
        except Exception:
            pass
    return _MODEL
