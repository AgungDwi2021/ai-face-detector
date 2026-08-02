import cv2
import mediapipe as mp

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from PIL import Image

import tensorflow as tf
import numpy as np
import sqlite3
import os
import json

from datetime import datetime

app = Flask(__name__)

# CONFIG

UPLOAD_FOLDER = 'static/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    'png',
    'jpg',
    'jpeg'
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('database', exist_ok=True)


# MEDIAPIPE FACE DETECTION
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7
)

# LOAD MODEL
interpreter = tf.lite.Interpreter(
    model_path="model/ai_face_detector.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# LOAD METRICS

with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)

# DATABASE

conn = sqlite3.connect(
    'database/predictions.db',
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    filename TEXT,

    prediction TEXT,

    confidence REAL,

    created_at TEXT
)
''')

conn.commit()

# VALIDATE FILE

def allowed_file(filename):

    return (
        '.' in filename and
        filename.rsplit(
            '.',
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )

# HUMAN FACE VALIDATION

def contains_face(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return False

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    results = face_detection.process(rgb)

    # NO FACE DETECTED

    if not results.detections:
        return False

    # CHECK DETECTION CONFIDENCE

    for detection in results.detections:

        confidence = detection.score[0]

        # ONLY STRONG HUMAN FACE

        if confidence > 0.7:
            return True

    return False

# PREPROCESS IMAGE

def preprocess_image(image_path):

    img = Image.open(image_path).convert('RGB')

    img = img.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array.astype(np.float32)

    # OPTIONAL NORMALIZATION

    # img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

# PREDICT IMAGE

def predict_image(image_path):

    input_data = preprocess_image(image_path)

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    score = float(prediction[0][0])
    print("Score:", score)
    
    # =====================================
    # LABEL
    # =====================================

    if score > 0.5:

        label = "REAL"

    else:

        label = "AI GENERATED"

    # =====================================
    # CONFIDENCE
    # =====================================

    confidence = round(
        max(score, 1 - score) * 100,
        2
    )

    return label, confidence

# HOME

@app.route(
    '/',
    methods=['GET', 'POST']
)

def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':

        # CHECK IMAGE

        if 'image' not in request.files:
            return redirect('/')

        file = request.files['image']

        if file.filename == '':
            return redirect('/')

        # VALID FILE

        if file and allowed_file(file.filename):

            filename = secure_filename(
                file.filename
            )

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            file.save(filepath)

            image_path = filepath

            # =================================
            # HUMAN FACE VALIDATION
            # =================================

            if not contains_face(filepath):

                prediction = "NO FACE DETECTED"

                confidence = 0

            else:

                # =============================
                # AI DETECTION
                # =============================

                prediction, confidence = predict_image(
                    filepath
                )

            # =================================
            # SAVE DATABASE
            # =================================

            cursor.execute('''
            INSERT INTO predictions
            (
                filename,
                prediction,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?)
            ''', (

                filename,

                prediction,

                confidence,

                datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                )
            ))

            conn.commit()

    # =====================================
    # RENDER TEMPLATE
    # =====================================

    return render_template(

        'index.html',

        accuracy=metrics['accuracy'],

        auc=metrics['auc_roc'],

        precision=metrics['precision'],

        recall=metrics['recall'],

        f1_score=metrics['f1_score'],

        prediction=prediction,

        confidence=confidence,

        image_path=image_path
    )

# HISTORY

@app.route('/history')

def history():

    cursor.execute('''
    SELECT *
    FROM predictions
    ORDER BY id DESC
    ''')

    data = cursor.fetchall()

    return render_template(
        'history.html',
        data=data
    )

# ABOUT

@app.route('/about')

def about():

    return render_template(
        'about.html'
    )

# ERROR FILE SIZE
@app.errorhandler(413)
def too_large(e):

    return render_template(
        'index.html',
        accuracy=metrics['accuracy'],
        auc=metrics['auc_roc'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1_score'],
        prediction=None,
        confidence=None,
        image_path=None,
        error="File terlalu besar, Maksimal 10 MB."
    ), 413

# RUN APP

if __name__ == '__main__':

    app.run(
        debug=True
    )