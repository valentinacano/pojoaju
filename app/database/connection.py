"""
Módulo de conexión a la base de datos PostgreSQL.

Este módulo proporciona una función para establecer una conexión
a la base de datos usando los parámetros definidos en la configuración
del proyecto (`DB_CONFIG`).
"""

import psycopg2
from app.config import DB_CONFIG


def get_connection():
    """
    Establece y retorna una conexión activa a la base de datos PostgreSQL.

    Utiliza la configuración definida en el archivo `.env` cargada en `DB_CONFIG`.
    Si ocurre un error al intentar conectar, imprime el error y vuelve a lanzar la excepción.

    Returns:
        psycopg2.extensions.connection: Objeto de conexión activa a la base de datos.

    Raises:
        Exception: Propaga cualquier excepción que ocurra durante la conexión.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print("❌ Error al conectar a la base de datos:", e)
        raise


