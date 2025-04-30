from app.database.connection import get_connection


def create_keypoints_table():
    """
    Crea la tabla `keypoints` en la base de datos si no existe.

    La tabla contiene columnas para:
    - ID autoincremental
    - ID de la palabra
    - Número de muestra
    - Frame dentro de la muestra
    - Datos de keypoints en formato JSONB
    - Timestamp automático

    Ejecuta una sentencia `CREATE TABLE IF NOT EXISTS`.

    Returns:
        None: Esta función no retorna ningún valor. Crea la tabla en la base de datos.

    Raises:
        Exception: Imprime y relanza cualquier error ocurrido durante la creación de la tabla.
    """
    query = """
    CREATE TABLE IF NOT EXISTS keypoints (
        id SERIAL PRIMARY KEY,
        word_id INT NOT NULL,
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
        print("❌ Error al crear la tabla:", e)
        raise
