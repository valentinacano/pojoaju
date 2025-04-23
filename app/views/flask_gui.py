#!/usr/bin/env python
"""
Aplicación web para captura y procesamiento de lenguaje de señas.

Esta aplicación permite al usuario capturar muestras de una palabra desde la cámara,
visualizar el video en tiempo real, y normalizar las muestras para generar keypoints
que serán usados en modelos de entrenamiento.

Incluye las siguientes rutas:
- Página de inicio
- Formulario de captura
- Captura de video por palabra
- Normalización de muestras capturadas
"""

from flask import Flask, render_template, Response, request
from ml.features.pipelines import (
    create_samples_from_camera,
    save_samples,
)
from app.config import FRAME_ACTIONS_PATH, KEYPOINTS_PATH

app = Flask(__name__)


@app.route("/")
def index():
    """
    Página principal del sitio.

    Returns:
        str: Render de la plantilla `index.html`.
    """
    return render_template("index.html")


@app.route("/training")
def training():
    """
    Página de entrenamiento del modelo.

    Returns:
        str: Render de la plantilla `training.html`.
    """
    return render_template("training.html")


@app.route("/capture_form")
def capture_form():
    """
    Página con formulario para ingresar una palabra y comenzar la captura.

    Returns:
        str: Render de la plantilla `capture_form.html`.
    """
    return render_template("capture_form.html")


@app.route("/capture/<word>")
def capture(word):
    """
    Página de captura para una palabra específica.

    Args:
        word (str): Palabra que se está capturando.

    Returns:
        str: Render de la plantilla `capture.html` con la palabra en contexto.
    """
    return render_template("capture.html", word=word)


@app.route("/video_feed/<word>")
def video_feed(word):
    """
    Provee un flujo de video en tiempo real con los keypoints dibujados.

    Args:
        word (str): Palabra que se está capturando.

    Returns:
        Response: Flujo continuo de frames JPEG para visualización en vivo.
    """
    return Response(
        create_samples_from_camera(word, FRAME_ACTIONS_PATH),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/save_samples/<word>")
def save_samples(word):
    """
    Normaliza las muestras capturadas y genera los keypoints en formato `.h5`.

    Args:
        word (str): Palabra que se desea normalizar.

    Returns:
        str: Render de la plantilla `save_samples.html` al completar el proceso.
    """
    save_samples(word, FRAME_ACTIONS_PATH, KEYPOINTS_PATH)
    return render_template("save_samples.html", word=word)
