from dotenv import load_dotenv
load_dotenv()  # ← DEBE ser antes de cualquier import de app/

from app.database.schema import create_all_tables
from app.database.queries import insert_words_bulk
from app.config import WORDS
from app.views.flask_gui import app


def initialize():
    """Crea tablas e inserta vocabulario inicial si no existe."""
    print("🛠️  Inicializando base de datos...")
    create_all_tables()
    insert_words_bulk(WORDS)
    print("✅ Base de datos lista.\n")


if __name__ == "__main__":
    initialize()
    app.run(debug=False)