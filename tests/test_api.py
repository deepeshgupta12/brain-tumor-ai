import io, numpy as np, cv2
from fastapi.testclient import TestClient
from app.main import app

def _fake_png_bytes():
    arr = (np.random.rand(256,256)*255).astype("uint8")
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return io.BytesIO(buf.tobytes())

def test_predict_endpoint():
    with TestClient(app) as client:
        files = {"file": ("dummy.png", _fake_png_bytes(), "image/png")}
        r = client.post("/predict", files=files)
        assert r.status_code == 200
        data = r.json()
        for k in ["has_tumor","tumor_probability","classification","segmentation_available","tumor_volume_ml","notes"]:
            assert k in data
        assert isinstance(data["has_tumor"], bool)
        assert 0.0 <= float(data["tumor_probability"]) <= 1.0
