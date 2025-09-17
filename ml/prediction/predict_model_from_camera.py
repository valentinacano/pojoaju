import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.database.database_utils import fetch_word_ids_with_keypoints, search_word_id
from ml.utils.keypoints_utils import mediapipe_detection, extract_keypoints
from ml.utils.common_utils import there_hand
from ml.utils.capture_utils import draw_keypoints
from app.config import MODEL_PATH, MODEL_FRAMES
from app.services.text_to_speech import text_to_speech

# ----- CONSTANTES
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.8
FONT_POS = (10, 30)
MIN_LENGTH_FRAMES = 10
PREDICTION_COOLDOWN = 15


def normalize_keypoints(keypoints, target_length=15):
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


def predict_model_from_camera(threshold=0.8):
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
                    cooldown_counter = PREDICTION_COOLDOWN  # bloqueo temporal

                recording = False
                kp_seq = []

            # Control del cooldown (para no predecir cada frame)
            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Mostrar resultado
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


if __name__ == "__main__":
    predict_model_from_camera()


# import cv2
# import numpy as np
# from mediapipe.python.solutions.holistic import Holistic
# from keras.models import load_model
# from ml.utils.keypoints_utils import mediapipe_detection, extract_keypoints
# from ml.utils.capture_utils import draw_keypoints
# from app.config import MODEL_PATH, MODEL_FRAMES


def predict_model_from_camera_stream(threshold=0.8):
    kp_seq, sentence = [], []
    model = load_model(MODEL_PATH)
    cooldown_counter = 0
    recording = False

    with Holistic() as holistic:
        cap = cv2.VideoCapture(2)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)

            # --- detección y predicción ---
            if results:
                kp_seq.append(extract_keypoints(results))
                recording = True
            elif recording:
                if len(kp_seq) >= 10 and cooldown_counter == 0:
                    # Normalización
                    normalized = kp_seq[:MODEL_FRAMES]  # simplificado
                    res = model.predict(np.expand_dims(normalized, axis=0))[0]

                    max_idx = np.argmax(res)
                    conf = res[max_idx]
                    predicted_word = f"Palabra {max_idx}"

                    if conf > threshold:
                        label = f"{predicted_word} ({conf*100:.2f}%) ✔️"
                    else:
                        label = f"{predicted_word} ({conf*100:.2f}%) ❌"

                    sentence.insert(0, label)
                    cooldown_counter = 15

                recording = False
                kp_seq = []

            if cooldown_counter > 0:
                cooldown_counter -= 1

            # --- overlay en el frame ---
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

            # --- encode frame ---
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        cap.release()

