# AI Generated Face Detector

A web application that detects whether a face image is **AI-generated** or **real** using **EfficientNet-B3**, **TensorFlow Lite**, **MediaPipe**, and **Flask**.

## Features

- Upload JPG, JPEG, or PNG images.
- Human face validation using MediaPipe.
- AI-generated vs. real face classification.
- Prediction confidence display.
- Prediction history stored in SQLite.

## Tech Stack

- Python
- Flask
- TensorFlow Lite
- MediaPipe
- OpenCV
- SQLite

## Installation

```bash
git clone https://github.com/Agungdwis12/face-detector-web.git
cd face-detector-web
pip install -r requirements.txt
python app.py
```

Open the application at:

```
http://127.0.0.1:5000
```

## Model Performance

- Accuracy: **90.83%**
- Precision: **86.19%**
- Recall: **97.25%**
- F1-Score: **91.39%**
- AUC-ROC: **97.25%**
