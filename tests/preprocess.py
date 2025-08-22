import numpy as np
from flujo_rubrica.preprocess_img import preprocess

def test_preprocess_output_shape_and_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)

    out = preprocess(img)

    assert out.shape == (1, 512, 512, 1)
    assert out.dtype in (np.float32, np.float64)
    assert np.isfinite(out).all()
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
