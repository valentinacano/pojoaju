#!/usr/bin/env python
from flask import Flask, render_template, Response, request
from ml.features.pipelines import create_samples_from_camera, normalize_samples_from_camera

app = Flask(__name__)
FRAME_ACTIONS_PATH = "data/frame_actions"
KEYPOINTS_PATH = "data/keypoints"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/capture_form')
def capture_form():
    return render_template('capture_form.html')

@app.route('/capture/<word>')
def capture(word):
    return render_template('capture.html', word=word)

@app.route('/video_feed/<word>')
def video_feed(word):
    return Response(
        create_samples_from_camera(word, FRAME_ACTIONS_PATH),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/normalize/<word>')
def normalize(word):
    normalize_samples_from_camera(word, FRAME_ACTIONS_PATH, KEYPOINTS_PATH)
    return render_template('normalize.html', word=word)