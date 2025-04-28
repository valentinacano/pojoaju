"""
Captura de muestras en video con detección de keypoints mediante MediaPipe.

Este módulo permite grabar muestras de lenguaje de señas desde la cámara web.
Detecta manos usando MediaPipe y guarda secuencias válidas de frames en carpetas
con timestamp para su posterior análisis o entrenamiento.
"""

import os
import cv2

from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic

from ml.utils.capture_utils import save_frames, draw_keypoints
from ml.utils.common_utils import create_folder, mediapipe_detection, there_hand
from app.config import FONT, FONT_POS, FONT_SIZE


def _save_sample(frames, path, margin_frames, delay_frames):
    """
    Guarda una muestra recortada (frames) en una carpeta con timestamp.

    Args:
        frames (list): Lista de frames capturados.
        path (str): Ruta donde se debe guardar la muestra.
        margin_frames (int): Frames a eliminar del inicio y final.
        delay_frames (int): Frames a eliminar del cierre.

    Returns:
        None
    """
    trimmed = frames[: -(margin_frames + delay_frames)]
    folder = os.path.join(path, f"sample_{datetime.now().strftime('%y%m%d%H%M%S%f')}")
    create_folder(folder)
    save_frames(trimmed, folder)


def capture_samples_from_camera(
    path, margin_frames=1, min_frames=5, delay_frames=3, debug=False, camera_index=0
):
    """
    Captura muestras desde la cámara y guarda las secuencias.

    Args:
        path (str): Ruta donde guardar las muestras.
        margin_frames (int): Frames a descartar al inicio y fin.
        min_frames (int): Mínimo de frames válidos requeridos.
        delay_frames (int): Frames adicionales antes de cortar la muestra.
        debug (bool): Si True, muestra ventana OpenCV. Si False, retorna frames JPEG.
        camera_index (int): Índice de la cámara (0 por defecto).

    Returns:
        generator | None: Si debug=False, retorna frames para streaming. Si debug=True, muestra ventana.
    """

    create_folder(path)
    frames, frame_count, fix_frames = [], 0, 0
    recording = False

    with Holistic() as model:
        cap = cv2.VideoCapture(camera_index)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, model)
            image = frame.copy()

            if there_hand(results) or recording:
                recording = False
                frame_count += 1
                if frame_count > margin_frames:
                    if debug:
                        cv2.putText(
                            image,
                            "Capturando...",
                            FONT_POS,
                            FONT,
                            FONT_SIZE,
                            (255, 50, 0),
                        )
                    frames.append(frame)
            else:
                if len(frames) >= min_frames + margin_frames:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    _save_sample(frames, path, margin_frames, delay_frames)

                recording, fix_frames, frames, frame_count = False, 0, [], 0
                if debug:
                    cv2.putText(
                        image,
                        "Listo para capturar...",
                        FONT_POS,
                        FONT,
                        FONT_SIZE,
                        (0, 220, 100),
                    )

            if debug:
                draw_keypoints(image, results)
                cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            else:
                draw_keypoints(image, results)
                ret, buffer = cv2.imencode(".jpg", image)
                frame = buffer.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        cap.release()
        if debug:
            cv2.destroyAllWindows()
