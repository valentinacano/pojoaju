"""
Predicción en tiempo real desde la cámara.

Puntos clave:
- Usa normalize_sequence() — MISMO método que train.py
- Carga el modelo una sola vez al iniciar el stream
- Soporta modo consola y modo Flask (streaming JPEG)
- TTS se ejecuta en hilo separado para no bloquear
- Acumula señas en un buffer para traducción de frases con Gemini
- Sin anotaciones en el video — toda la info va al panel HTML
"""

import cv2
import json
import numpy as np
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic

from app.config import MODEL_PATH, MODEL_FRAMES, PREDICTION_THRESHOLD, PREDICTION_COOLDOWN, MIN_FRAMES_CAPTURED
from app.database.queries import fetch_word_ids_with_keypoints, get_word_by_id
from app.services.text_to_speech import text_to_speech_async
from ml.keypoints import run_mediapipe, extract_keypoints, has_hand
from ml.capture import _draw_landmarks

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_POS = (10, 35)
FONT_SIZE = 0.8

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------

_current_phrase = []       # palabras acumuladas para traducción
_last_prediction = {}      # última predicción: {word, conf, accepted}


def get_current_phrase() -> list[str]:
    """Retorna las señas acumuladas en la frase actual."""
    return _current_phrase.copy()


def clear_phrase():
    """Limpia el buffer de la frase actual."""
    global _current_phrase
    _current_phrase = []


def get_last_prediction() -> dict:
    """Retorna la última predicción realizada."""
    return _last_prediction.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_idx_to_word(word_ids: list) -> dict:
    idx_to_word = {}
    for i, wid in enumerate(word_ids):
        row = get_word_by_id(bytes(wid))
        if row:
            idx_to_word[i] = row[1]
        else:
            idx_to_word[i] = f"clase_{i}"
    return idx_to_word


def _predict_sequence(model, sequence: list, idx_to_word: dict) -> tuple[str, float]:
    from ml.normalize import normalize_sequence
    normalized = normalize_sequence(sequence, MODEL_FRAMES)
    X = np.expand_dims(normalized, axis=0).astype(np.float32)
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    word = idx_to_word.get(idx, f"clase_{idx}")
    return word, conf


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def predict_stream(camera_index: int = 0):
    """
    Generador de predicción en tiempo real para Flask.

    El video se muestra limpio — sin barras ni texto superpuesto.
    Toda la información (palabra, porcentaje, frase) va al panel HTML.
    """
    global _current_phrase, _last_prediction

    word_ids = fetch_word_ids_with_keypoints()
    idx_to_word = _build_idx_to_word(word_ids)
    model = load_model(MODEL_PATH)

    kp_seq = []
    recording = False
    cooldown = 0

    with Holistic() as holistic:
        cap = cv2.VideoCapture(camera_index)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = run_mediapipe(frame, holistic)

            if has_hand(results):
                kp_seq.append(extract_keypoints(results))
                recording = True

            elif recording:
                if len(kp_seq) >= MIN_FRAMES_CAPTURED and cooldown == 0:
                    word, conf = _predict_sequence(model, kp_seq, idx_to_word)
                    accepted = conf >= PREDICTION_THRESHOLD

                    # Guardar última predicción para el frontend
                    _last_prediction = {
                        "word": word,
                        "conf": round(conf * 100, 1),
                        "accepted": accepted
                    }

                    if accepted:
                        text_to_speech_async(word)
                        _current_phrase.append(word)

                    cooldown = PREDICTION_COOLDOWN

                recording = False
                kp_seq = []

            if cooldown > 0:
                cooldown -= 1

            # Video limpio — solo landmarks, sin barras ni texto
            _draw_landmarks(frame, results)

            _, buffer = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

        cap.release()


def predict_console(camera_index: int = 0, threshold: float = PREDICTION_THRESHOLD):
    """
    Predicción en tiempo real en consola. Presionar 'q' para salir.
    """
    global _current_phrase, _last_prediction

    word_ids = fetch_word_ids_with_keypoints()
    idx_to_word = _build_idx_to_word(word_ids)
    model = load_model(MODEL_PATH)

    kp_seq = []
    recording = False
    cooldown = 0

    with Holistic() as holistic:
        cap = cv2.VideoCapture(camera_index)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = run_mediapipe(frame, holistic)

            if has_hand(results):
                kp_seq.append(extract_keypoints(results))
                recording = True

            elif recording:
                if len(kp_seq) >= MIN_FRAMES_CAPTURED and cooldown == 0:
                    word, conf = _predict_sequence(model, kp_seq, idx_to_word)
                    accepted = conf >= threshold

                    _last_prediction = {
                        "word": word,
                        "conf": round(conf * 100, 1),
                        "accepted": accepted
                    }

                    if accepted:
                        text_to_speech_async(word)
                        _current_phrase.append(word)
                        print(f"✔ {word} ({conf*100:.1f}%)")
                    else:
                        print(f"✗ {word} ({conf*100:.1f}%)")

                    cooldown = PREDICTION_COOLDOWN

                recording = False
                kp_seq = []

            if cooldown > 0:
                cooldown -= 1

            _draw_landmarks(frame, results)
            cv2.imshow("Pojoaju — Predicción", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()