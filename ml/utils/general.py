"""
Utilidades generales para detección con MediaPipe, manejo de carpetas
y lectura de archivos JSON.

Estas funciones se utilizan en el proceso de reconocimiento de lenguaje de señas
para facilitar tareas comunes como el preprocesamiento de imágenes, verificación
de manos detectadas y carga de vocabulario desde archivos externos.
"""

import json
import os
import cv2


def mediapipe_detection(image, model):
    """
    Ejecuta el modelo de MediaPipe sobre una imagen dada.

    Convierte la imagen de BGR a RGB, la marca como no editable para mejorar el rendimiento
    y la pasa al modelo para obtener los resultados de detección.

    Args:
        image (numpy.ndarray): Imagen en formato BGR capturada desde la cámara.
        model: Modelo de MediaPipe ya inicializado y cargado.

    Returns:
        NamedTuple: Resultado devuelto por el modelo, que incluye los landmarks detectados.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results


def create_folder(path):
    """
    Crea una carpeta en la ruta especificada si no existe.

    Verifica si la carpeta ya existe en el sistema de archivos. Si no existe, la crea;
    en caso contrario, no realiza ninguna acción adicional.

    Args:
        path (str): Ruta completa donde se desea crear la carpeta.

    Returns:
        None: Esta función no retorna ningún valor.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def there_hand(results):
    """
    Verifica si hay al menos una mano presente en los resultados del modelo.

    Examina los landmarks detectados en la mano izquierda o derecha. Si alguno de ellos
    está presente en los resultados, se considera que una mano fue detectada.

    Args:
        results (NamedTuple): Objeto de resultados devuelto por el modelo de MediaPipe.

    Returns:
        bool: True si se detecta al menos una mano; False si no se detecta ninguna.
    """
    return results.left_hand_landmarks or results.right_hand_landmarks


def get_word_ids(path):
    """
    Obtiene una lista de IDs de palabras desde un archivo JSON.

    Abre el archivo JSON especificado, lo convierte a un diccionario y busca la clave "word_ids".
    Si la clave está presente, devuelve la lista correspondiente; de lo contrario, retorna None.

    Args:
        path (str): Ruta del archivo JSON que contiene las IDs de palabras.

    Returns:
        list | None: Lista de IDs de palabras si la clave está presente, o None si no existe.
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)
        return data.get("word_ids")
