"""
Captura de muestras desde archivo de video con detección de keypoints.

Este módulo permite procesar un archivo de video previamente grabado y detectar
frames relevantes mediante MediaPipe Holistic. Cuando se detecta actividad válida
(presencia de manos), se almacenan los frames en una carpeta organizada por muestra.

Incluye:
- `_save_sample`: guarda la secuencia de frames como imágenes numeradas en una subcarpeta.
- `capture_samples_from_video`: procesa frame por frame el video, detecta actividad y guarda muestras válidas.

Usos comunes:
- Entrenamiento offline desde grabaciones
- Ingesta de muestras externas para el pipeline
"""

import os, cv2

from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic

from ml.utils.capture_utils import save_frames, draw_keypoints
from ml.utils.common_utils import create_folder, mediapipe_detection, there_hand
from app.config import FONT, FONT_POS, FONT_SIZE


def _save_sample(frames, path, margin_frames, delay_frames):
    """
    Guarda una muestra de frames recortada en una carpeta con timestamp.

    Corta los frames del final (según los márgenes indicados), crea una carpeta
    con nombre basado en la hora actual y guarda las imágenes como archivos .jpg.

    Args:
        frames (list[np.ndarray]): Lista de frames capturados.
        path (str): Carpeta donde se deben guardar las muestras.
        margin_frames (int): Cantidad de frames descartados al inicio.
        delay_frames (int): Cantidad de frames descartados al final.

    Returns:
        None: Los archivos son guardados en disco, no se retorna valor.
    """
    trimmed = frames[: -(margin_frames + delay_frames)]
    folder = os.path.join(path, f"sample_{datetime.now().strftime('%y%m%d%H%M%S%f')}")
    create_folder(folder)
    save_frames(trimmed, folder)


def capture_samples_from_video(
    video_path, path, margin_frames=1, min_frames=5, delay_frames=3, debug=False
):
    """
    Captura muestras de lenguaje de señas a partir de un video previamente grabado.

    Detecta la presencia de manos en cada frame usando MediaPipe Holistic. Si detecta
    actividad válida, empieza a grabar una muestra. Al finalizar, guarda la secuencia
    de frames como imágenes numeradas en una carpeta por muestra.

    En modo `debug=True`, muestra los frames en tiempo real con anotaciones.

    Args:
        video_path (str): Ruta al archivo de video a procesar.
        path (str): Carpeta base donde guardar las muestras.
        margin_frames (int, optional): Frames a descartar al inicio y fin de la muestra. Default: 1.
        min_frames (int, optional): Mínimo de frames válidos requeridos para guardar una muestra. Default: 5.
        delay_frames (int, optional): Cantidad de frames de retardo antes de cortar una muestra. Default: 3.
        debug (bool, optional): Si es True, muestra el procesamiento en una ventana OpenCV. Default: False.

    Returns:
        None: Procesa los frames y guarda las muestras en disco.
    """
    frames, frame_count, fix_frames = [], 0, 0
    recording = False

    # Abrimos el vídeo en lugar de la cámara
    cap = cv2.VideoCapture(video_path)
    with Holistic() as model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, model)
            display_img = frame.copy()

            if there_hand(results) or recording:
                recording = False
                frame_count += 1
                if frame_count > margin_frames:
                    if debug:
                        cv2.putText(
                            display_img,
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

                # reset
                recording, fix_frames, frames, frame_count = False, 0, [], 0
                if debug:
                    cv2.putText(
                        display_img,
                        "Listo para capturar...",
                        FONT_POS,
                        FONT,
                        FONT_SIZE,
                        (0, 220, 100),
                    )

            # dibujar y mostrar o descartar
            draw_keypoints(display_img, results)
            if debug:
                cv2.imshow(
                    f'Procesando vídeo "{os.path.basename(video_path)}"', display_img
                )
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        cap.release()
        if debug:
            cv2.destroyAllWindows()
