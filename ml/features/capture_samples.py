"""
Captura de muestras en video con detección de keypoints mediante MediaPipe.

Este módulo permite grabar muestras de lenguaje de señas desde la cámara web.
Utiliza MediaPipe Holistic para detectar manos en tiempo real, y guarda
automáticamente las secuencias válidas de frames en carpetas numeradas con timestamp.

Se puede usar en dos modos:
- Modo consola (`debug=True`): muestra el proceso con OpenCV.
- Modo servidor (`debug=False`): genera frames JPEG para transmitir por Flask (`video_feed`).

Esta funcionalidad es utilizada por la app web (ruta `/video_feed/<word>`) y forma
parte del flujo de entrenamiento en vivo desde cámara.
"""

import os, cv2

from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic

from ml.utils.capture_utils import save_frames, draw_keypoints
from ml.utils.common_utils import create_folder, mediapipe_detection, there_hand
from app.config import FONT, FONT_POS, FONT_SIZE


stop_capture = False  # Usado por la ruta Flask '/stop_capture' para detener la grabación en tiempo real


def _save_sample(frames, path, margin_frames, delay_frames):
    """
    Guarda una muestra recortada (frames) en una carpeta con timestamp.

    Corta los frames excedentes del inicio y final, genera una carpeta con
    nombre único (timestamp) y guarda las imágenes como archivos individuales.

    Si la secuencia no tiene suficientes frames luego del recorte, no se guarda nada.

    Args:
        frames (list[np.ndarray]): Lista de frames capturados.
        path (str): Ruta donde se debe guardar la muestra.
        margin_frames (int): Cantidad de frames descartados del inicio.
        delay_frames (int): Cantidad de frames descartados del cierre.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    trimmed = frames[: -(margin_frames + delay_frames)]
    if len(trimmed) == 0:
        print("⚠️ La muestra recortada está vacía. No se guardará nada.")
        return

    folder = os.path.join(path, f"sample_{datetime.now().strftime('%y%m%d%H%M%S%f')}")
    create_folder(folder)
    save_frames(trimmed, folder)


def capture_samples_from_camera(
    path, margin_frames=1, min_frames=5, delay_frames=3, debug=False, camera_index=0
):
    """
    Captura muestras desde la cámara y guarda las secuencias válidas.

    Utiliza MediaPipe Holistic para detectar presencia de manos. Cuando se detecta
    actividad válida, comienza a grabar. Si no hay detección por una cierta cantidad
    de frames (`delay_frames`), guarda la muestra si cumple con el mínimo de frames.

    Puede ejecutarse en dos modos:
    - `debug=True`: visualiza el proceso en una ventana OpenCV.
    - `debug=False`: retorna un generador de imágenes JPEG para streaming (ej. Flask).

    Args:
        path (str): Ruta donde se guardarán las muestras.
        margin_frames (int, optional): Frames a descartar al inicio y fin. Default: 1.
        min_frames (int, optional): Mínimo de frames válidos requeridos. Default: 5.
        delay_frames (int, optional): Frames adicionales antes de cortar la muestra. Default: 3.
        debug (bool, optional): Modo visual. Si True, muestra ventana OpenCV. Default: False.
        camera_index (int, optional): Índice del dispositivo de cámara. Default: 0.

    Returns:
        generator | None:
            - En modo Flask (`debug=False`): generador de imágenes JPEG.
            - En modo consola (`debug=True`): no retorna nada.
    """

    global stop_capture

    create_folder(path)
    frames, frame_count, fix_frames = [], 0, 0
    recording = False

    with Holistic() as model:
        cap = cv2.VideoCapture(camera_index)

        while cap.isOpened():
            if stop_capture:  # Detener la captura desde Flask
                break

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

        stop_capture = False  # Se resetea al finalizar la captura
