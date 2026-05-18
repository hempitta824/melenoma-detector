"""
Download the pre-trained skin lesion model (one-time setup).

Run: python download_model.py
"""

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/syaha/skin_cancer_detection_model/resolve/main/"
    "skin_cancer_model.h5"
)
MODEL_PATH = Path("models") / "skin_cancer_model.h5"


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        print(f"Model already exists at {MODEL_PATH}")
        return

    print("Downloading model (~128 MB). This may take a few minutes...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
