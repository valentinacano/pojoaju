"""
Predicción en tiempo real desde la cámara.

Puntos clave:
- Usa normalize_sequence() — MISMO método que train.py
- Carga el modelo una sola vez al iniciar el stream
- Soporta modo consola y modo Flask (streaming JPEG)
- TTS se ejecuta en hilo separado para no bloquear
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
# Helpers
# ---------------------------------------------------------------------------

def _build_idx_to_word(word_ids: list) -> dict:
    """
    Construye un diccionario {índice: nombre_de_palabra}.

    Args:
        word_ids: lista de word_ids ordenada (mismo orden que el training).

    Returns:
        dict {int: str}
    """
    idx_to_word = {}
    for i, wid in enumerate(word_ids):
        row = get_word_by_id(bytes(wid))
        if row:
            idx_to_word[i] = row[1]  # row = (word_id, word, category)
        else:
            idx_to_word[i] = f"clase_{i}"
    return idx_to_word


def _predict_sequence(model, sequence: list, idx_to_word: dict) -> tuple[str, float]:
    """
    Predice la palabra a partir de una secuencia de keypoints.

    Usa normalize_sequence() — idéntico al training.

    Args:
        model: modelo Keras cargado.
        sequence: lista de np.ndarray (keypoints por frame).
        idx_to_word: mapeo índice → nombre de palabra.

    Returns:
        (palabra_predicha, confianza)
    """
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
    Generador de predicción en tiempo real para Flask (modo streaming).

    Captura frames de la cámara, acumula keypoints mientras hay manos
    detectadas, predice al soltar y retorna JPEG anotados.

    Args:
        camera_index: índice de la cámara (default 0).

    Yields:
        bytes JPEG anotados con la predicción.
    """
    word_ids = fetch_word_ids_with_keypoints()
    idx_to_word = _build_idx_to_word(word_ids)
    model = load_model(MODEL_PATH)

    kp_seq = []
    sentence = []
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

                    if conf >= PREDICTION_THRESHOLD:
                        label = f"{word} ({conf*100:.1f}%) ✔"
                        text_to_speech_async(word)
                    else:
                        label = f"{word} ({conf*100:.1f}%) ✗"

                    sentence.insert(0, label)
                    sentence = sentence[:3]
                    cooldown = PREDICTION_COOLDOWN

                recording = False
                kp_seq = []

            if cooldown > 0:
                cooldown -= 1

            # Anotar frame
            cv2.rectangle(frame, (0, 0), (640, 50), (245, 117, 16), -1)
            cv2.putText(frame, " | ".join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255), 2)
            _draw_landmarks(frame, results)

            _, buffer = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

        cap.release()


def predict_console(camera_index: int = 0, threshold: float = PREDICTION_THRESHOLD):
    """
    Ejecuta predicción en tiempo real mostrando una ventana OpenCV.

    Presionar 'q' para salir.

    Args:
        camera_index: índice de la cámara.
        threshold: umbral de confianza mínima para aceptar predicción.
    """
    word_ids = fetch_word_ids_with_keypoints()
    idx_to_word = _build_idx_to_word(word_ids)
    model = load_model(MODEL_PATH)

    kp_seq = []
    sentence = []
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

                    if conf >= threshold:
                        label = f"{word} ({conf*100:.1f}%) ✔"
                        text_to_speech_async(word)
                    else:
                        label = f"{word} ({conf*100:.1f}%) ✗"

                    sentence.insert(0, label)
                    sentence = sentence[:3]
                    cooldown = PREDICTION_COOLDOWN

                recording = False
                kp_seq = []

            if cooldown > 0:
                cooldown -= 1

            cv2.rectangle(frame, (0, 0), (640, 50), (245, 117, 16), -1)
            cv2.putText(frame, " | ".join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255), 2)
            _draw_landmarks(frame, results)
            cv2.imshow("Pojoaju — Predicción", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()