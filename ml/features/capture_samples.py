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
    path, margin_frames=1, min_frames=5, delay_frames=3, debug=False
):
    """
    Captura automáticamente muestras de video al detectar manos en cámara.

    Esta función utiliza la cámara web para detectar manos mediante MediaPipe Holistic.
    Cuando se detectan manos, comienza a grabar frames. Una vez que se deja de detectar,
    verifica si la muestra tiene la cantidad mínima de frames válidos y la guarda.
    Puede funcionar en modo visual (`debug=True`) o como generador de imágenes JPEG en tiempo real.

    Args:
        path (str): Ruta donde se guardarán las muestras capturadas.
        margin_frames (int): Cantidad de frames a descartar al inicio y fin de la muestra.
        min_frames (int): Cantidad mínima de frames necesarios para que una muestra sea válida.
        delay_frames (int): Frames adicionales a grabar antes de finalizar la muestra al perder detección de manos.
        debug (bool): Si es True, se muestra la imagen con los keypoints en pantalla.
                      Si es False, se comporta como generador de imágenes codificadas JPEG para streaming.

    Returns:
        None: Si debug=True, no retorna nada y muestra los frames en pantalla.
        Generator[bytes]: Si debug=False, retorna un generador de imágenes JPEG para transmisión en vivo.
    """
    print(f"\n📸 Iniciando captura de muestras en: {path}")
    frames, frame_count, fix_frames = [], 0, 0
    recording = False
    with Holistic() as model:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = mediapipe_detection(frame, model)
            image = frame.copy()

            if there_hand(results) or recording:
                recording = False
                frame_count += 1
                if frame_count > margin_frames:
                    cv2.putText(
                        image, "Capturando...", FONT_POS, FONT, FONT_SIZE, (255, 50, 0)
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
                cv2.putText(
                    image,
                    "Listo para capturar...",
                    FONT_POS,
                    FONT,
                    FONT_SIZE,
                    (0, 220, 100),
                )

            draw_keypoints(image, results)
            if debug:
                cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            else:
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        video.release()
        cv2.destroyAllWindows()