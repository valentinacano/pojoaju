"""
Aplicación principal del sistema de reconocimiento de señas.

Inicializa las tablas de la base de datos y lanza la aplicación Flask.
La inicialización se realiza una única vez para evitar duplicación con el reloader.
"""

import os
from app.config import words, categories
from app.views.flask_gui import app
from app.database.schema import (
    create_categories_table,
    create_words_table,
    create_keypoints_table,
    create_samples_table,
)
from app.database.database_utils import (
    insert_words,
    insert_categories,
)

from ml.training.training_model import training_model
from ml.features.pipelines import create_samples_from_video
from ml.prediction.predict_model_from_camera import predict_model_from_camera

from app.config import FRAME_ACTIONS_PATH, VIDEO_EXPORT_PATH


def initialize_database():
    """
    Crea las tablas necesarias e inserta palabras y categorías si no existen.

    Esta función debe ejecutarse una sola vez al inicio del sistema para preparar
    la base de datos. No borra datos previos.

    Returns:
        None
    """
    print("🛠️ Inicializando base de datos (sin borrar datos existentes)...")
    create_categories_table()
    create_words_table()
    create_samples_table()
    create_keypoints_table()
    insert_categories(categories)
    insert_words(words)
    print("✅ Base de datos lista.\n")


if __name__ == "__main__":
    # initialize_database()
    #app.run(debug=True)
    
    #training_model()
    predict_model_from_camera()
# create_samples_from_video(
#    word_name="papá",
#    video_path=VIDEO_EXPORT_PATH,
#    root_path=FRAME_ACTIONS_PATH
# )
