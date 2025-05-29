from app.database.connection import get_connection


def create_keypoints_table():
    """
    Crea la tabla `keypoints` en la base de datos si no existe.

    Esta tabla almacena los vectores de keypoints extraídos de las muestras,
    incluyendo información sobre la palabra, muestra y frame.

    Columnas:
    - `keypoints_id` (SERIAL PRIMARY KEY): Identificador único del registro.
    - `word_id` (INT): Identificador de la palabra.
    - `sample` (INT): Número de la muestra.
    - `frame` (INT): Número del frame dentro de la muestra.
    - `keypoints` (JSONB): Vectores de keypoints en formato JSON.
    - `timestamp` (TIMESTAMP): Fecha y hora de creación del registro (por defecto `NOW()`).

    Returns:
        None: Esta función no retorna ningún valor.

    Raises:
        Exception: Lanza cualquier error que ocurra durante la creación de la tabla.
    """
    query = """
    CREATE TABLE IF NOT EXISTS keypoints (
        keypoints_id SERIAL PRIMARY KEY,
        word_id BYTEA NOT NULL,
        sample INT NOT NULL,
        frame INT NOT NULL,
        keypoints JSONB NOT NULL,
        timestamp TIMESTAMP DEFAULT NOW()
    );
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Tabla 'keypoints' creada (o ya existía).")
    except Exception as e:
        print("❌ Error al crear la tabla 'keypoints':", e)
        raise


def create_categories_table():
    """
    Crea la tabla `categories` en la base de datos si no existe.

    Esta tabla almacena las categorías a las que pertenecen las palabras
    del sistema de reconocimiento de lenguaje de señas.

    Columnas:
    - `category_id` (SERIAL PRIMARY KEY): Identificador único de la categoría.
    - `category` (VARCHAR): Nombre de la categoría (único).
    - `created_at` (TIMESTAMP): Fecha y hora de creación del registro (por defecto `NOW()`).
    - `updated_at` (TIMESTAMP): Fecha y hora de la última actualización del registro (por defecto `NOW()`).

    Returns:
        None: Esta función no retorna ningún valor.

    Raises:
        Exception: Lanza cualquier error que ocurra durante la creación de la tabla.
    """
    query = """
    CREATE TABLE IF NOT EXISTS categories (
        category_id SERIAL PRIMARY KEY,
        category VARCHAR(50) NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Tabla 'categories' creada (o ya existía).")
    except Exception as e:
        print("❌ Error al crear la tabla 'categories':", e)
        raise


def create_words_table():
    """
    Crea la tabla `words` en la base de datos si no existe.

    Esta tabla almacena las palabras capturadas y asociadas a categorías
    en el sistema de reconocimiento de lenguaje de señas.

    Columnas:
    - `word_id` (BYTEA PRIMARY KEY): Identificador único de la palabra (hash).
    - `category_id` (INT): Identificador de la categoría (FK).
    - `word` (VARCHAR): Texto de la palabra (único).
    - `created_at` (TIMESTAMP): Fecha y hora de creación del registro (por defecto `NOW()`).
    - `updated_at` (TIMESTAMP): Fecha y hora de la última actualización del registro (por defecto `NOW()`).

    Returns:
        None: Esta función no retorna ningún valor.

    Raises:
        Exception: Lanza cualquier error que ocurra durante la creación de la tabla.
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
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Tabla 'words' creada (o ya existía).")
    except Exception as e:
        print("❌ Error al crear la tabla 'words':", e)
        raise
