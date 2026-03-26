"""
Creación del esquema de base de datos.
"""

from app.database.connection import get_connection


def _run(query: str, label: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ {label}")


def create_all_tables():
    """Crea todas las tablas necesarias si no existen."""

    _run("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id SERIAL PRIMARY KEY,
            category    VARCHAR(100) NOT NULL UNIQUE,
            created_at  TIMESTAMP DEFAULT NOW()
        );
    """, "Tabla 'categories' lista")

    _run("""
        CREATE TABLE IF NOT EXISTS words (
            word_id     BYTEA PRIMARY KEY,
            category_id INT NOT NULL REFERENCES categories(category_id),
            word        VARCHAR(100) NOT NULL UNIQUE,
            created_at  TIMESTAMP DEFAULT NOW()
        );
    """, "Tabla 'words' lista")

    _run("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id  SERIAL PRIMARY KEY,
            word_id    BYTEA NOT NULL REFERENCES words(word_id),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """, "Tabla 'samples' lista")

    _run("""
        CREATE TABLE IF NOT EXISTS keypoints (
            keypoint_id SERIAL PRIMARY KEY,
            sample_id   INT NOT NULL REFERENCES samples(sample_id),
            word_id     BYTEA NOT NULL REFERENCES words(word_id),
            frame       INT NOT NULL,
            keypoints   JSONB NOT NULL,
            created_at  TIMESTAMP DEFAULT NOW()
        );
    """, "Tabla 'keypoints' lista")