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

    Convierte la imagen de BGR a RGB, desactiva la escritura para mejorar el rendimiento
    y aplica el modelo para obtener los resultados de detección.

    Args:
        image (np.ndarray): Imagen en formato BGR.
        model: Modelo de MediaPipe (por ejemplo, Holistic()).

    Returns:
        NamedTuple: Resultados devueltos por el modelo, incluyendo landmarks detectados.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return model.process(image)


def create_folder(path):
    """
    Crea una carpeta en la ruta especificada si no existe.

    Si la carpeta ya existe, no realiza ninguna acción.

    Args:
        path (str): Ruta completa de la carpeta a crear.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def there_hand(results):
    """
    Verifica si hay al menos una mano detectada en los resultados de MediaPipe.

    Args:
        results (NamedTuple): Objeto de resultados devuelto por MediaPipe.

    Returns:
        bool: True si se detecta al menos una mano, False si no.
    """
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def get_word_ids(json_path):
    """
    Extrae una lista de IDs de palabras desde un archivo JSON.

    Intenta leer el archivo JSON y obtener la clave `word_ids`. Si falla,
    imprime un mensaje de advertencia y retorna None.

    Args:
        json_path (str): Ruta al archivo JSON que contiene los IDs.

    Returns:
        list | None: Lista de IDs si la clave existe, o None si ocurre un error o no está presente.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("word_ids")
    except Exception as e:
        print(f"⚠️  Error al leer archivo JSON: {e}")
        return None
