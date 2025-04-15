#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import threading
from ml.features.capture_samples import generate_frames
import cv2

app = Flask(__name__)
FRAME_ACTIONS_PATH = "data/frame_actions"

@app.route('/capture/<word>')
def capture(word):
    return render_template('capture.html', word=word)

@app.route('/video_feed/<word>')
def video_feed(word):
    return Response(
        generate_frames(word, FRAME_ACTIONS_PATH),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
