import os

from ml.features.capture_samples import capture_samples
from ml.features.create_keypoints import create_keypoints
from ml.utils.general import create_folder
from app.config import ROOT_PATH, FRAME_ACTIONS_PATH, KEYPOINTS_PATH, words_text


if __name__ == "__main__":
    accion = "keyword"

    if accion == "capture samples":
        word_name = "buenos_dias"
        word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
        capture_samples(word_path)
    else:
        # Crea la carpeta `keypoints` en caso no exista
        create_folder(KEYPOINTS_PATH)

        # GENERAR TODAS LAS PALABRAS
        word_ids = words_text
        word_ids = [
            word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))
        ]

        # GENERAR PARA UNA PALABRA O CONJUNTO
        # word_ids = ["bien"]
        # word_ids = ["buenos_dias", "como_estas", "disculpa", "gracias", "hola-der", "hola-izq", "mal", "mas_o_menos", "me_ayudas", "por_favor"]

        for word_id in word_ids:
            hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
            create_keypoints(word_id, FRAME_ACTIONS_PATH, hdf_path)
