"""
Captura de muestras en video con detección de keypoints usando MediaPipe.

Este módulo permite grabar muestras de lenguaje de señas desde la cámara web.
Detecta la presencia de manos mediante MediaPipe y guarda secuencias válidas
de frames en carpetas numeradas para su posterior análisis y entrenamiento.
"""

import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from ml.utils.capture_samples import save_frames, draw_keypoints
from ml.utils.general import create_folder, mediapipe_detection, there_hand
from app.config import FONT, FONT_POS, FONT_SIZE
from datetime import datetime


def capture_samples(path, margin_frame=1, min_cant_frames=5, delay_frames=3):
    """
    Captura muestras de una palabra o gesto desde la cámara web.

    Graba automáticamente una secuencia de frames cuando se detectan manos usando MediaPipe.
    Agrega un margen de inicio y fin, y guarda los frames como imágenes en una carpeta con timestamp.

    Args:
        path (str): Ruta a la carpeta donde se guardarán las muestras.
        margin_frame (int): Cantidad de frames que se ignoran al comienzo y al final de la captura.
        min_cant_frames (int): Cantidad mínima de frames requeridos para guardar una muestra.
        delay_frames (int): Cantidad de frames adicionales que se graban antes de finalizar la muestra cuando ya no se detectan manos.

    Returns:
        None: Esta función no retorna ningún valor. Guarda los archivos en disco.
    """
    create_folder(path)

    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)

            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(
                        image, "Capturando...", FONT_POS, FONT, FONT_SIZE, (255, 50, 0)
                    )
                    frames.append(np.asarray(frame))
            else:
                if len(frames) >= min_cant_frames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: -(margin_frame + delay_frames)]
                    today = datetime.now().strftime("%y%m%d%H%M%S%f")
                    output_folder = os.path.join(path, f"sample_{today}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)

                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(
                    image,
                    "Listo para capturar...",
                    FONT_POS,
                    FONT,
                    FONT_SIZE,
                    (0, 220, 100),
                )

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()
