"""
Predicción de palabras en lenguaje de señas desde la cámara utilizando modelos LSTM.

Este módulo permite capturar keypoints en tiempo real desde la cámara,
normalizarlos a una longitud fija y realizar predicciones con un modelo LSTM
entrenado previamente.

Incluye dos formas de ejecución:
- `predict_model_from_camera`: para ejecución en consola con OpenCV.
- `predict_model_from_camera_stream`: para streaming desde Flask.

Además, se incluye una función de texto a voz (`text_to_speech`) para reproducir
la palabra reconocida mediante audio. Se ejecuta en segundo plano con
`text_to_speech_async` para evitar bloqueo de la interfaz web.
"""

import os
import cv2
import numpy as np
import threading
import tempfile

from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from gtts import gTTS
from playsound import playsound

from app.database.database_utils import fetch_word_ids_with_keypoints, search_word_id
from app.config import MODEL_PATH, MODEL_FRAMES
from app.services.text_to_speech import text_to_speech
from ml.utils.keypoints_utils import mediapipe_detection, extract_keypoints
from ml.utils.common_utils import there_hand
from ml.utils.capture_utils import draw_keypoints

# ----- CONSTANTES
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.8
FONT_POS = (10, 30)
MIN_LENGTH_FRAMES = 10
PREDICTION_COOLDOWN = 15


def normalize_keypoints(keypoints, target_length=15):
    """
    Ajusta una secuencia de keypoints a una longitud fija por interpolación o recorte.

    Si la secuencia tiene menos frames que `target_length`, interpola los valores
    para completar. Si tiene más, recorta uniformemente.

    Args:
        keypoints (list[list[float]]): Lista de frames con keypoints.
        target_length (int): Longitud deseada de la secuencia.

    Returns:
        list[list[float]]: Secuencia ajustada a la longitud requerida.
    """
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    elif current_length < target_length:
        indices = np.linspace(0, current_length - 1, target_length)
        interpolated = []
        for i in indices:
            low, high = int(np.floor(i)), int(np.ceil(i))
            weight = i - low
            point = (
                (1 - weight) * np.array(keypoints[low])
                + weight * np.array(keypoints[high])
                if low != high
                else np.array(keypoints[low])
            )
            interpolated.append(point.tolist())
        return interpolated
    else:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]


def predict_model_from_camera(threshold=0.5):
    """
    Ejecuta el flujo de predicción desde cámara en consola.

    Captura secuencias de keypoints usando MediaPipe, las normaliza y las
    pasa por el modelo LSTM para predecir la palabra. El resultado se muestra
    en pantalla y se reproduce con texto a voz si la confianza supera el umbral.

    Args:
        threshold (float, optional): Umbral de confianza para aceptar la predicción. Default: 0.5.

    Returns:
        list[str]: Lista de las últimas palabras reconocidas (máximo 3).
    """
    kp_seq, sentence = [], []

    word_ids = fetch_word_ids_with_keypoints()
    idx_to_word = {}
    for i, word_id in enumerate(word_ids):
        result = search_word_id(word_id)
        if result:
            _, word, _ = result
            idx_to_word[i] = word

    model = load_model(MODEL_PATH)
    recording = False
    cooldown_counter = 0

    with Holistic() as holistic:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)

            if there_hand(results):
                kp_seq.append(extract_keypoints(results))
                recording = True
            elif recording:
                if len(kp_seq) >= MIN_LENGTH_FRAMES and cooldown_counter == 0:
                    normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    res = model.predict(np.expand_dims(normalized, axis=0))[0]

                    max_idx = np.argmax(res)
                    conf = res[max_idx]
                    predicted_word = idx_to_word.get(max_idx, "<desconocido>")

                    if conf > threshold:
                        label = f"{predicted_word} ({conf * 100:.2f}%) ✔️"
                        text_to_speech(predicted_word)
                    else:
                        label = f"{predicted_word} ({conf * 100:.2f}%) ❌"

                    sentence.insert(0, label)
                    cooldown_counter = PREDICTION_COOLDOWN

                recording = False
                kp_seq = []

            if cooldown_counter > 0:
                cooldown_counter -= 1

            cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
            cv2.putText(
                frame,
                " | ".join(sentence[:3]),
                FONT_POS,
                FONT,
                FONT_SIZE,
                (255, 255, 255),
            )
            draw_keypoints(frame, results)
            cv2.imshow("Traductor LSP", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()
        return sentence


def text_to_speech(text):
    """
    Convierte un texto en audio y lo reproduce.

    Utiliza gTTS para generar el archivo de voz y lo reproduce con playsound.

    Args:
        text (str): Texto a pronunciar.

    Returns:
        None
    """
    print(f"🔊 DICIENDO: {text}")
    tts = gTTS(text=text, lang="es")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        playsound(fp.name)
        os.unlink(fp.name)


def text_to_speech_async(text):
    """
    Ejecuta `text_to_speech` en segundo plano.

    Esto permite que la interfaz Flask no se bloquee mientras se reproduce
    el audio.

    Args:
        text (str): Texto a pronunciar.

    Returns:
        None
    """
    threading.Thread(target=text_to_speech, args=(text,)).start()


def predict_model_from_camera_stream(threshold=0.8):
    """
    Ejecuta la predicción desde cámara en modo streaming Flask.

    Captura los keypoints desde la cámara, realiza predicciones usando el modelo
    entrenado y genera un flujo continuo de imágenes con anotaciones y etiquetas
    predichas para renderizar en tiempo real desde la interfaz web.

    Args:
        threshold (float, optional): Umbral de confianza para aceptar la predicción. Default: 0.8.

    Yields:
        bytes: Imágenes JPEG codificadas para streaming tipo multipart.
    """
    kp_seq, sentence = [], []
    model = load_model(MODEL_PATH)
    cooldown_counter = 0
    recording = False

    word_ids = fetch_word_ids_with_keypoints()
    idx_to_word = {}
    for i, word_id in enumerate(word_ids):
        result = search_word_id(word_id)
        if result:
            _, word, _ = result
            idx_to_word[i] = word

    with Holistic() as holistic:
        cap = cv2.VideoCapture(0)  # Cambiar a 0 si usás cámara interna

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)

            if there_hand(results):
                kp_seq.append(extract_keypoints(results))
                recording = True
            elif recording:
                if len(kp_seq) >= MODEL_FRAMES and cooldown_counter == 0:
                    normalized = kp_seq[:MODEL_FRAMES]
                    res = model.predict(np.expand_dims(normalized, axis=0))[0]

                    max_idx = np.argmax(res)
                    conf = res[max_idx]
                    predicted_word = idx_to_word.get(max_idx, f"Palabra {max_idx}")

                    if conf > threshold:
                        label = f"{predicted_word} ({conf*100:.2f}%) ✔️"
                        text_to_speech_async(predicted_word)
                    else:
                        label = f"{predicted_word} ({conf*100:.2f}%) ❌"

                    sentence.insert(0, label)
                    cooldown_counter = PREDICTION_COOLDOWN

                recording = False
                kp_seq = []

            if cooldown_counter > 0:
                cooldown_counter -= 1

            cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
            cv2.putText(
                frame,
                " | ".join(sentence[:3]),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            draw_keypoints(frame, results)

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        cap.release()


if __name__ == "__main__":
    predict_model_from_camera()
