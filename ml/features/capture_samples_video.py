import os, cv2
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic

from ml.utils.capture_utils import save_frames, draw_keypoints
from ml.utils.common_utils import create_folder, mediapipe_detection, there_hand
from app.config import FONT, FONT_POS, FONT_SIZE

def _save_sample(frames, path, margin_frames, delay_frames):
    trimmed = frames[: -(margin_frames + delay_frames)]
    folder = os.path.join(path, f"sample_{datetime.now().strftime('%y%m%d%H%M%S%f')}")
    create_folder(folder)
    save_frames(trimmed, folder)

def capture_samples_from_video(
    video_path,
    output_path,
    margin_frames=1,
    min_frames=5,
    delay_frames=3,
    debug=False
):
    """
    Igual que capture_samples_from_camera pero para un fichero de vídeo.
    
    Args:
        video_path (str): Ruta al archivo de vídeo.
        output_path (str): Carpeta donde guardar las muestras.
        margin_frames (int): Frames a descartar al inicio y fin.
        min_frames (int): Mínimo de frames válidos requeridos.
        delay_frames (int): Frames de retardo antes de guardar.
        debug (bool): Si True, muestra ventana con OpenCV.
    """
    create_folder(output_path)
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
                        cv2.putText(display_img,
                                    "Capturando...",
                                    FONT_POS, FONT, FONT_SIZE,
                                    (255,50,0))
                    frames.append(frame)
            else:
                if len(frames) >= min_frames + margin_frames:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    _save_sample(frames, output_path, margin_frames, delay_frames)

                # reset
                recording, fix_frames, frames, frame_count = False, 0, [], 0
                if debug:
                    cv2.putText(display_img,
                                "Listo para capturar...",
                                FONT_POS, FONT, FONT_SIZE,
                                (0,220,100))

            # dibujar y mostrar o descartar
            draw_keypoints(display_img, results)
            if debug:
                cv2.imshow(f'Procesando vídeo "{os.path.basename(video_path)}"', display_img)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        cap.release()
        if debug:
            cv2.destroyAllWindows()
