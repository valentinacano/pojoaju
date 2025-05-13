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

from flask import Flask, render_template, Response
from ml.features.pipelines import (
    create_samples_from_camera,
    save_keypoints,
)
from app.database.database_utils import fetch_all_words
from app.config import FRAME_ACTIONS_PATH

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


@app.route("/training/capture/<word_id>/<word>")
def capture(word_id, word):
    """
    Página de captura para una palabra específica.

    Args:
        word (str): Palabra que se está capturando.

    Returns:
        str: Render de la plantilla `capture.html` con la palabra en contexto.
    """
    return render_template("capture.html", word_id=word_id, word=word)


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
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/save_samples/<word>/<word_id>")
def save_samples(word, word_id):
    """
    Normaliza las muestras capturadas y extrae los keypoints para una palabra.

    Este endpoint ejecuta el pipeline completo de procesamiento:
    - Normaliza los frames capturados
    - Extrae los keypoints usando MediaPipe Holistic
    - Guarda los resultados en la base de datos (tabla `keypoints`)

    Args:
        word (str): Palabra que fue capturada y debe procesarse.
        word_id (str): ID correspondiente a la palabra en la base de datos.

    Returns:
        str: Render de la plantilla `save_samples.html` al completar el proceso.
    """
    save_keypoints(word, word_id, FRAME_ACTIONS_PATH)
    return render_template("save_samples.html", word=word, word_id=word_id)


def filter_words(filter_text, words):
    filter_text = filter_text.strip().lower()
    return [
        word
        for word in words
        if filter_text in word[1].lower() or filter_text in word[2].lower()
    ]


@app.route("/training/dictionary")
def dictionary():
    words = fetch_all_words()
    return render_template("dictionary.html", words=words)


@app.route("/training/dictionary/search", methods=["POST"])
def dictionary_search():
    filter_text = request.form.get("filter")
    words = fetch_all_words()
    filtered_words = filter_words(filter_text, words)
    return render_template("dictionary.html", words=filtered_words)
