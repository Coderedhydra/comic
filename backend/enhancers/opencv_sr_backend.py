import os
from typing import Optional


MODEL_URL = "https://github.com/Saafke/EDSR_TensorFlow/releases/download/1.0/EDSR_x2.pb"


def _ensure_model(model_dir: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "EDSR_x2.pb")
    if not os.path.isfile(model_path):
        import requests
        resp = requests.get(MODEL_URL, timeout=30)
        resp.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(resp.content)
    return model_path


def enhance_with_opencv_sr(input_path: str, output_path: str, scale: int = 2) -> bool:
    try:
        import cv2
        from cv2 import dnn_superres
        model_path = _ensure_model(os.path.join(os.path.dirname(__file__), "models"))
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", scale)
        img = cv2.imread(input_path)
        if img is None:
            return False
        up = sr.upsample(img)
        cv2.imwrite(output_path, up)
        return True
    except Exception:
        return False

