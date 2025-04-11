"""
Funciones generales de soporte para detección con MediaPipe, manejo de carpetas
y lectura de archivos JSON.

Estas utilidades se utilizan durante el preprocesamiento, detección y organización
de los datos en el sistema de reconocimiento de señas.
"""

import os
import json
import cv2


def mediapipe_detection(image, model):
    """
    Ejecuta la detección de MediaPipe sobre una imagen.

    Convierte de BGR a RGB, desactiva escritura para rendimiento,
    ejecuta el modelo y devuelve los resultados.

    Args:
        image (np.ndarray): Imagen en formato BGR.
        model: Modelo de MediaPipe (e.g., Holistic()).

    Returns:
        NamedTuple: Resultados del modelo (landmarks detectados).
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return model.process(image)


def create_folder(path):
    """
    Crea una carpeta si no existe.

    Args:
        path (str): Ruta deseada.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def there_hand(results):
    """
    Verifica si hay al menos una mano detectada.

    Args:
        results (NamedTuple): Resultado de MediaPipe.

    Returns:
        bool: True si hay landmarks de mano izquierda o derecha.
    """
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def get_word_ids(json_path):
    """
    Extrae la lista de IDs de palabras desde un archivo JSON.

    Args:
        json_path (str): Ruta al archivo JSON.

    Returns:
        list | None: Lista de IDs si existe, o None.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("word_ids")
    except Exception as e:
        print(f"⚠️  Error al leer archivo JSON: {e}")
        return None
