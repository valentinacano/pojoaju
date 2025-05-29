import os
from app.config import words, categories
from app.views.flask_gui import app
from app.database.schema import (
    create_categories_table,
    create_words_table,
    create_keypoints_table,
)
from app.database.database_utils import (
    insert_words,
    insert_categories,
)


def initialize_database():
    print("🛠️ Inicializando base de datos (sin borrar datos existentes)...")
    create_categories_table()
    create_words_table()
    create_keypoints_table()
    insert_categories(categories)
    insert_words(words)
    print("✅ Base de datos lista.\n")


if __name__ == "__main__":
    # Evita la ejecución duplicada del reloader de Flask
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        initialize_database()

    app.run(debug=True)
