import hashlib, pytest, psycopg2
import numpy as np

from app.database.connection import get_connection
from app.database.schema import create_all_tables
from app.database.database_utils import (
    insert_categories,
    insert_words,
    insert_sample,
    insert_keypoints,
    fetch_keypoints_by_words,
    fetch_word_ids_with_keypoints,
    fetch_all_words,
    fetch_all_categories,
    search_word,
)
from app.config import words, categories, DB_CONFIG


# -------------------- TEST CONEXIÓN A BD --------------------


def test_database_connection():
    """
    Verifica que se establezca una conexión activa a la base de datos
    y que corresponda al nombre esperado en `DB_CONFIG`.

    Returns:
        None
    """
    conn = get_connection()
    dbname_expected = "pojoaju_test"
    dbname_actual = conn.get_dsn_parameters().get("dbname")

    assert conn is not None, "La conexión retornada es None"
    assert conn.closed == 0, "La conexión está cerrada"
    assert (
        dbname_actual == dbname_expected
    ), f"Se esperaba conexión a '{dbname_expected}', pero se conectó a '{dbname_actual}'"
    if dbname_actual != "pojoaju_test":
        pytest.exit("❌ Conectado a una base que no es de pruebas. Deteniendo tests.")
    conn.close()


# -------------------- TEST ESTRUCTURA DE BD --------------------


def get_existing_tables():
    """
    Recupera los nombres de las tablas existentes en la base de datos actual.

    Returns:
        list[str]: Lista con nombres de tablas en el esquema 'public'.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public';
    """
    )
    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return tables


def test_schema_tables_created(clean_test_database):
    """
    Verifica que todas las tablas requeridas estén creadas correctamente.

    Este test ejecuta `create_all_tables()` y luego revisa si existen las tablas
    `categories`, `words`, `samples` y `keypoints` en el esquema `public`.

    Returns:
        None
    """
    create_all_tables()
    expected_tables = {"categories", "words", "samples", "keypoints"}
    existing_tables = set(get_existing_tables())

    missing_tables = expected_tables - existing_tables
    assert not missing_tables, f"Faltan tablas: {missing_tables}"


# -------------------- FIXTURE PARA CARGA DE DATOS --------------------


@pytest.fixture(scope="function")
def setup_test_schema(clean_test_database):
    """
    Crea las tablas y carga los datos iniciales (palabras y categorías) en la base de pruebas.

    Returns:
        None
    """
    create_all_tables()
    insert_categories(categories)
    insert_words(words)


# -------------------- TEST FUNCIONAL --------------------


def test_fetch_all_words_and_categories(setup_test_schema):
    all_words = fetch_all_words()
    all_categories = fetch_all_categories()

    assert len(all_categories) == len(categories)
    assert len(all_words) > 0
    assert any("hola" in row for row in [w[1] for w in all_words])


def test_insert_sample_and_keypoints(setup_test_schema):
    word = "hola"
    word_id = hashlib.sha256(word.encode("utf-8")).digest()
    sample_id = insert_sample(word_id)

    dummy_keypoints = [np.random.rand(1662) for _ in range(5)]
    insert_keypoints(word_id, sample_id, dummy_keypoints)

    results = fetch_keypoints_by_words([word_id])
    assert len(results) == 5

    stored_word_ids = [bytes(wid) for wid in fetch_word_ids_with_keypoints()]
    assert word_id in stored_word_ids


def test_search_word_found(setup_test_schema):
    result = search_word("hola")
    assert result is not None
    assert result[1] == "hola"


def test_search_word_not_found(setup_test_schema):
    result = search_word("NoExiste")
    assert result is None
