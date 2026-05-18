# Melanoma Detector

A beginner-friendly Flask web app that estimates whether a skin lesion image looks like **melanoma** or **benign**, with a confidence percentage.

> **Disclaimer:** This project is for learning only. It is not a medical device. Always consult a dermatologist for real diagnosis.

## What you need

- **Python 3.10, 3.11, 3.12, or 3.13** (TensorFlow does not support 3.14+ yet)
- About 2 GB free disk space (TensorFlow + model weights)

> If `pip install` says *No matching distribution found for tensorflow*, your venv was likely created with Python 3.14+. Recreate it with 3.13 (see step 1 below).

## Project layout

```
melenoma-detector/
├── app.py              # Flask server and prediction logic
├── download_model.py   # One-time download of the AI model
├── requirements.txt    # Python packages
├── templates/
│   └── index.html      # Web page (upload form + results)
└── models/             # Created after download (not in git)
    └── skin_cancer_model.h5
```

## Setup (step by step)

### 1. Create a virtual environment (recommended)

Use **Python 3.13** if you have multiple versions installed (Windows `py` launcher):

```powershell
cd melenoma-detector
# Remove old venv if it was created with Python 3.14
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
py -3.13 -m venv venv
```

Or with a single Python 3.10–3.13 install:

```bash
cd melenoma-detector
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, Pillow, NumPy, and TensorFlow. The first install may take several minutes.

### 3. Download the AI model (one time)

```bash
python download_model.py
```

This downloads a pre-trained model (~128 MB) trained on the [HAM10000](https://www.nature.com/articles/sdata2018161) skin lesion dataset.

### 4. Run the app

```bash
python app.py
```

Open your browser to: **http://127.0.0.1:5000**

Upload a JPG or PNG image of a skin lesion and click **Analyze image**.

## How it works (simple overview)

1. **Upload** — The browser sends the image to Flask (`app.py`).
2. **Preview** — The image is encoded as base64 and shown on the page.
3. **Preprocess** — The image is resized to 224×224 pixels and normalized (values 0–1).
4. **Predict** — A Keras neural network outputs probabilities; we compare melanoma vs. non-melanoma.
5. **Display** — The page shows **Melanoma** or **Benign** and a confidence percentage.

Key functions in `app.py`:

| Function | Purpose |
|----------|---------|
| `get_model()` | Loads the `.h5` model file once |
| `preprocess_image()` | Prepares the image for the network |
| `predict()` | Returns label and confidence |
| `index()` | Flask route for the home page |

## Tips for beginners

- Change the port in `app.py`: `app.run(debug=True, port=5000)`
- Set `debug=False` before sharing the app on a network.
- Use dermatoscopic-style lesion images for more realistic results.
- If you see “Model not found”, run `python download_model.py` again.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not found` | Run `python download_model.py` |
| Slow first prediction | TensorFlow loads the model on first use — wait a few seconds |
| `pip` not found | Use `python -m pip install -r requirements.txt` |
| Upload too large | Max file size is 8 MB (set in `app.py`) |

## License

Educational use. Model weights from [syaha/skin_cancer_detection_model](https://huggingface.co/syaha/skin_cancer_detection_model) on Hugging Face.
