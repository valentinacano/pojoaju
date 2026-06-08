"""
Captura de muestras de lenguaje de señas.

Soporta dos fuentes:
- Cámara en vivo (modo consola y modo Flask streaming)
- Archivo de video pregrabado
"""

import os
import cv2
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.holistic import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    FACEMESH_CONTOURS,
)
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

from ml.keypoints import run_mediapipe, has_hand
from app.config import (
    MARGIN_FRAMES,
    MIN_FRAMES_SAMPLE,
    DELAY_FRAMES,
    FONT,
    FONT_POS,
    FONT_SIZE,
)

_stop_capture = False


def stop_capture():
    global _stop_capture
    _stop_capture = True


def _draw_landmarks(image, results):
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def _save_sample(frames: list, path: str):
    """Guarda una secuencia de frames como muestra en disco."""
    # Recortar solo si hay suficientes frames para el margen
    if len(frames) > MARGIN_FRAMES + DELAY_FRAMES:
        trimmed = frames[: -(MARGIN_FRAMES + DELAY_FRAMES)]
    else:
        trimmed = frames

    if len(trimmed) == 0:
        print("⚠️ Muestra vacía luego del recorte, se descarta.")
        return

    folder = os.path.join(path, f"sample_{datetime.now().strftime('%y%m%d%H%M%S%f')}")
    os.makedirs(folder, exist_ok=True)

    for i, frame in enumerate(trimmed, start=1):
        cv2.imwrite(
            os.path.join(folder, f"frame_{i:02d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

    print(f"💾 Muestra guardada: {os.path.basename(folder)} ({len(trimmed)} frames)")


def capture_from_camera(path: str, debug: bool = False, camera_index: int = 0):
    """
    Captura muestras desde la cámara web.

    El Holistic se mantiene abierto durante todo el generador
    para evitar el error '_graph is None'.
    """
    global _stop_capture
    _stop_capture = False
    os.makedirs(path, exist_ok=True)

    frames = []
    frame_count = 0
    fix_frames = 0
    recording = False

    cap = cv2.VideoCapture(0)

    with Holistic() as holistic:
        while cap.isOpened():
            if _stop_capture:
                break

            ret, frame = cap.read()
            if not ret:
                break

            results = run_mediapipe(frame, holistic)
            display = frame.copy()

            if has_hand(results) or recording:
                recording = False
                frame_count += 1
                if frame_count > MARGIN_FRAMES:
                    frames.append(frame.copy())
                    if debug:
                        cv2.putText(
                            display,
                            "Capturando...",
                            FONT_POS,
                            FONT,
                            FONT_SIZE,
                            (255, 50, 0),
                        )
            else:
                if len(frames) >= MIN_FRAMES_SAMPLE + MARGIN_FRAMES:
                    fix_frames += 1
                    if fix_frames < DELAY_FRAMES:
                        recording = True
                    else:
                        _save_sample(frames, path)
                        frames, frame_count, fix_frames, recording = [], 0, 0, False
                else:
                    frames, frame_count, fix_frames, recording = [], 0, 0, False
                    if debug:
                        cv2.putText(
                            display,
                            "Listo...",
                            FONT_POS,
                            FONT,
                            FONT_SIZE,
                            (0, 220, 100),
                        )

            _draw_landmarks(display, results)

            if debug:
                cv2.imshow("Captura", display)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            else:
                ret, buffer = cv2.imencode(".jpg", display)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    # Guardar lo que quedó acumulado al cerrar
    if len(frames) >= MIN_FRAMES_SAMPLE:
        _save_sample(frames, path)

    cap.release()
    _stop_capture = False
    if debug:
        cv2.destroyAllWindows()


def capture_from_video(video_path: str, path: str, sample_count: int = 1):
    """
    Procesa el video completo como UNA seña y la replica
    `sample_count` veces para generar múltiples muestras.
    """
    os.makedirs(path, exist_ok=True)

    # 1. Extraer todos los frames del video una sola vez
    cap = cv2.VideoCapture(video_path)
    all_frames = []

    with Holistic() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = run_mediapipe(frame, holistic)

            if has_hand(results):
                all_frames.append(frame.copy())

    cap.release()

    if len(all_frames) < MIN_FRAMES_SAMPLE:
        print(
            f"⚠️  No se detectaron suficientes frames con mano ({len(all_frames)}). Muestra descartada."
        )
        return

    print(f"📹 Video procesado: {len(all_frames)} frames válidos detectados")

    # 2. Guardar la misma seña N veces
    for i in range(sample_count):
        _save_sample(all_frames.copy(), path)
        print(f"✅ Muestra {i + 1}/{sample_count} guardada")

    print(f"✅ {sample_count} muestras guardadas.")
