import json
import os
import cv2
from typing import NamedTuple


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results


def create_folder(path):
    """
    ### CREAR CARPETA SI NO EXISTE
    Si ya existe, no hace nada.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks


def get_word_ids(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
        return data.get("word_ids")
