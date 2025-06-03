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
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        initialize_database()
    app.run(debug=True)
