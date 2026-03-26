"""
Conexión a PostgreSQL.
"""

import psycopg2
from app.config import DB_CONFIG


def get_connection():
    """
    Retorna una conexión activa a PostgreSQL usando DB_CONFIG.

    Returns:
        psycopg2.connection

    Raises:
        Exception: si la conexión falla.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        raise