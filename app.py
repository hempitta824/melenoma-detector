"""
Melanoma Skin Lesion Detector — Flask web app.

Upload a skin lesion image; the app predicts melanoma vs benign
and shows a confidence score.
"""

import base64
import io
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

MODEL_PATH = Path("models") / "skin_cancer_model.h5"
# Index 6 is melanoma in the HAM10000-trained 7-class model
MELANOMA_CLASS_INDEX = 6

_model = None


def get_model():
    """Load the Keras model once and reuse it for every request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run: python download_model.py"
            )
        from tensorflow.keras.models import load_model

        _model = load_model(MODEL_PATH)
    return _model


def preprocess_image(image_bytes):
    """Resize and normalize the image for the model (224×224 RGB)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_bytes):
    """Return label ('Melanoma' or 'Benign') and confidence (0–100)."""
    model = get_model()
    batch = preprocess_image(image_bytes)
    probs = model.predict(batch, verbose=0)[0]
    mel_prob = float(probs[MELANOMA_CLASS_INDEX])

    if mel_prob >= 0.5:
        return "Melanoma", round(mel_prob * 100, 1)
    return "Benign", round((1.0 - mel_prob) * 100, 1)


def image_to_base64(image_bytes):
    """Encode uploaded bytes so the template can show a preview."""
    return base64.b64encode(image_bytes).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_data = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please choose an image file."
        else:
            image_bytes = file.read()
            if not image_bytes:
                error = "That file appears to be empty."
            else:
                try:
                    image_data = image_to_base64(image_bytes)
                    result, confidence = predict(image_bytes)
                except FileNotFoundError as exc:
                    error = str(exc)
                except Exception:
                    error = (
                        "Could not process that image. "
                        "Use a JPG or PNG of a skin lesion."
                    )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_data=image_data,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
