# ASL Sign Language Recognition

This repository contains a trained model and prediction scripts for recognizing
American Sign Language (ASL) letters using MediaPipe hand landmarks.

## Files

* `predict.py` - simple command-line script that uses your webcam to display
  predictions in a window.
* `dash_app.py` - interactive Plotly Dash web interface.  It supports two modes:
  - **Upload Image**: drop an image file containing a hand sign and see the
    predicted letter along with a skeleton overlay.
  - **Webcam**: stream your webcam directly into the browser with live
    predictions and landmark rendering.
* `asl_sign_model.h5` - trained Keras model.
* `scaler.pkl` / `label_encoder.pkl` - preprocessing artifacts used by the
  model.
* `hand_landmarker.task` - MediaPipe model used by the Python code.
* other helper scripts/datasets used during training.

## Requirements

Install dependencies in a Python environment (venv, conda, etc):

```powershell
python -m pip install dash opencv-python mediapipe tensorflow joblib pillow
```

> **Note:** the versions of `tensorflow` and `mediapipe` must be compatible with
> each other and your platform.  If you run into installation problems, refer to
> the respective project documentation.

## Running the Dash Interface

From the workspace root execute:

```powershell
python dash_app.py
```

A local web server will start (by default at `http://127.0.0.1:8050`).  Open that
URL in a browser to interact with the application.

Use the radio buttons at the top to switch between uploading a static image and
seeing live webcam predictions.

## Running the Original Predict Script

```powershell
python predict.py
```

This will open a window using OpenCV and show the live camera feed along with
predicted sign and confidence value.

---

Feel free to modify the Dash layout to add charts, histories or other
Plotly-powered interactive elements.  The prediction logic is contained in the
`annotate_and_predict` helper in `dash_app.py` and can be extracted into a
shared module if you need to avoid duplication.