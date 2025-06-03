from app.database.connection import get_connection


def _execute_query(query: str, table_name: str):
    """
    Ejecuta una consulta SQL para crear una tabla en la base de datos.

    Esta función encapsula la lógica de conexión, ejecución y confirmación
    para crear una tabla, mostrando el estado del proceso por consola.

    Args:
        query (str): Consulta SQL para la creación de tabla.
        table_name (str): Nombre de la tabla para mostrar en consola.

    Returns:
        None: No retorna ningún valor.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Tabla '{table_name}' creada (o ya existía).")
    except Exception as e:
        print(f"❌ Error al crear la tabla '{table_name}':", e)
        raise


def create_categories_table():
    """
    Crea la tabla `categories` si no existe.

    Almacena las categorías a las que pertenece cada palabra capturada.

    Columnas:
    - `category_id` (SERIAL): ID autoincremental de la categoría.
    - `category` (VARCHAR): Nombre único de la categoría.
    - `created_at`, `updated_at` (TIMESTAMP): Tiempos de registro.

    Returns:
        None
    """
    query = """
    CREATE TABLE IF NOT EXISTS categories (
        category_id SERIAL PRIMARY KEY,
        category VARCHAR(50) NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    _execute_query(query, "categories")


def create_words_table():
    """
    Crea la tabla `words` si no existe.

    Almacena las palabras reconocidas, asociadas a una categoría.

    Columnas:
    - `word_id` (BYTEA): ID único hash de la palabra.
    - `category_id` (INT): Clave foránea a `categories`.
    - `word` (VARCHAR): Texto de la palabra.
    - `created_at`, `updated_at` (TIMESTAMP): Tiempos de registro.

    Returns:
        None
    """
    query = """
    CREATE TABLE IF NOT EXISTS words (
        word_id BYTEA PRIMARY KEY,
        category_id INT NOT NULL REFERENCES categories(category_id),
        word VARCHAR(100) NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    _execute_query(query, "words")


def create_samples_table():
    """
    Crea la tabla `samples` en la base de datos si no existe.

    Esta tabla almacena los identificadores de cada muestra por palabra.

    Columnas:
    - `sample_id` (SERIAL PRIMARY KEY): Identificador único de la muestra.
    - `word_id` (BYTEA): Identificador de la palabra asociada (FK).
    - `created_at` (TIMESTAMP): Fecha de inserción automática.
    """
    query = """
    CREATE TABLE IF NOT EXISTS samples (
        sample_id SERIAL PRIMARY KEY,
        word_id BYTEA NOT NULL REFERENCES words(word_id),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    _execute_query(query, "samples")


def create_keypoints_table():
    """
    Crea la tabla `keypoints` si no existe.

    Almacena los vectores de keypoints extraídos por frame.

    Columnas:
    - `keypoints_id` (SERIAL): ID del frame.
    - `sample_id` (INT): Clave foránea a `samples`.
    - `word_id` (BYTEA): Clave foránea a `words`.
    - `frame` (INT): Número del frame en la muestra.
    - `keypoints` (JSONB): Coordenadas de keypoints.
    - `timestamp` (TIMESTAMP): Fecha de creación.

    Returns:
        None
    """
    query = """
    CREATE TABLE IF NOT EXISTS keypoints (
        keypoints_id SERIAL PRIMARY KEY,
        sample_id INT NOT NULL REFERENCES samples(sample_id),
        word_id BYTEA NOT NULL REFERENCES words(word_id),
        frame INT NOT NULL,
        keypoints JSONB NOT NULL,
        timestamp TIMESTAMP DEFAULT NOW()
    );
    """
    _execute_query(query, "keypoints")


def create_all_tables():
    """
    Ejecuta la creación de todas las tablas necesarias para el sistema.

    Crea las tablas `categories`, `words`, `samples` y `keypoints`
    de forma secuencial y segura.

    Returns:
        None
    """
    create_categories_table()
    create_words_table()
    create_samples_table()
    create_keypoints_table()
