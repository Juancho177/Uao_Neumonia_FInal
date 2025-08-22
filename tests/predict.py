import numpy as np
import tensorflow as tf
from flujo_rubrica import integrator as itg

class DummyModel:
    def __init__(self, out):
        self._out = tf.convert_to_tensor(out, dtype=tf.float32)
    def __call__(self, x, training=False):
        return self._out

def test_predict_returns_label_proba_and_heatmap(monkeypatch):
    dummy_out = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
    monkeypatch.setattr(itg, "model_fun", lambda: DummyModel(dummy_out), raising=True)

    def fake_grad_cam(array_bgr, **kwargs):
        H, W = array_bgr.shape[:2]
        return np.zeros((H, W, 3), dtype=np.uint8)
    monkeypatch.setattr(itg, "grad_cam", fake_grad_cam, raising=True)

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(520, 520, 3), dtype=np.uint8)

    label, proba, heat = itg.predict(img)

    assert label == "normal"
    assert np.isclose(proba, 80.0, atol=1e-5)
    assert heat.shape[2] == 3
