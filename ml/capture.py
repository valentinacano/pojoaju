"""
Captura de muestras de lenguaje de señas.

Soporta dos fuentes:
- Cámara en vivo (modo consola y modo Flask streaming)
- Archivo de video pregrabado

Ambas funciones usan la misma lógica de detección y guardado.
"""

import os
import cv2
import shutil
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.holistic import HAND_CONNECTIONS, POSE_CONNECTIONS, FACEMESH_CONTOURS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

from ml.keypoints import run_mediapipe, has_hand
from app.config import MARGIN_FRAMES, MIN_FRAMES_SAMPLE, DELAY_FRAMES, FONT, FONT_POS, FONT_SIZE


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _draw_landmarks(image, results):
    """Dibuja todos los landmarks sobre el frame."""
    draw_landmarks(image, results.face_landmarks, FACEMESH_CONTOURS,
                   DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                   DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    draw_landmarks(image, results.pose_landmarks, POSE_CONNECTIONS,
                   DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                   DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    draw_landmarks(image, results.left_hand_landmarks, HAND_CONNECTIONS,
                   DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                   DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    draw_landmarks(image, results.right_hand_landmarks, HAND_CONNECTIONS,
                   DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                   DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def _save_sample(frames: list, path: str):
    """
    Guarda una secuencia de frames como muestra en disco.

    Recorta los márgenes, crea una carpeta con timestamp y guarda
    cada frame como .jpg numerado.

    Args:
        frames: lista de frames capturados.
        path: carpeta raíz donde guardar la muestra.
    """
    trimmed = frames[:-(MARGIN_FRAMES + DELAY_FRAMES)]
    if len(trimmed) == 0:
        print("⚠️ Muestra vacía luego del recorte, se descarta.")
        return

    folder = os.path.join(path, f"sample_{datetime.now().strftime('%y%m%d%H%M%S%f')}")
    os.makedirs(folder, exist_ok=True)

    for i, frame in enumerate(trimmed, start=1):
        cv2.imwrite(os.path.join(folder, f"frame_{i:02d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"💾 Muestra guardada: {os.path.basename(folder)} ({len(trimmed)} frames)")


def _capture_loop(cap, model, path: str, debug: bool, stop_flag=None):
    """
    Loop principal de captura compartido entre cámara y video.

    Detecta presencia de manos, acumula frames y guarda muestras válidas.

    Args:
        cap: VideoCapture de OpenCV.
        model: instancia activa de Holistic.
        path: carpeta donde guardar las muestras.
        debug: si True muestra ventana OpenCV; si False genera JPEG para streaming.
        stop_flag: función que retorna True cuando se debe detener (opcional).

    Yields:
        bytes JPEG (solo en modo Flask, debug=False).
    """
    frames = []
    frame_count = 0
    fix_frames = 0
    recording = False

    while cap.isOpened():
        if stop_flag and stop_flag():
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = run_mediapipe(frame, model)
        display = frame.copy()

        if has_hand(results) or recording:
            recording = False
            frame_count += 1
            if frame_count > MARGIN_FRAMES:
                frames.append(frame.copy())
                if debug:
                    cv2.putText(display, "Capturando...", FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
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
                    cv2.putText(display, "Listo...", FONT_POS, FONT, FONT_SIZE, (0, 220, 100))

        _draw_landmarks(display, results)

        if debug:
            cv2.imshow("Captura", display)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            ret, buffer = cv2.imencode(".jpg", display)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    cap.release()
    if debug:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

# Flag global para detener el stream desde Flask
_stop_capture = False


def stop_capture():
    """Señala al generador de captura que debe detenerse."""
    global _stop_capture
    _stop_capture = True


def capture_from_camera(path: str, debug: bool = False, camera_index: int = 0):
    """
    Captura muestras desde la cámara web.

    Args:
        path: carpeta donde guardar las muestras.
        debug: True = ventana OpenCV, False = generador JPEG para Flask.
        camera_index: índice de la cámara (default 0).

    Returns:
        Generador de JPEG si debug=False, None si debug=True.
    """
    global _stop_capture
    _stop_capture = False
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)

    with Holistic() as model:
        gen = _capture_loop(cap, model, path, debug, stop_flag=lambda: _stop_capture)
        if debug:
            for _ in gen:
                pass
            _stop_capture = False
        else:
            return gen


def capture_from_video(video_path: str, path: str, debug: bool = False):
    """
    Captura muestras desde un archivo de video pregrabado.

    Args:
        video_path: ruta al archivo de video (.mp4, .mov, .avi).
        path: carpeta donde guardar las muestras.
        debug: True = ventana OpenCV, False = procesa silenciosamente.
    """
    os.makedirs(path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    with Holistic() as model:
        for _ in _capture_loop(cap, model, path, debug=False):
            pass

    print(f"✅ Video procesado: {video_path}")