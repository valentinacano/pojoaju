import psycopg2


def test_connection():
    try:
        # Conexión a la base de datos
        conn = psycopg2.connect(
            dbname="pojoaju",
            user="valentinacano",
            password="",  # Si no tiene clave, podés dejarlo vacío o directamente no usar el parámetro
            host="localhost",
            port="5432",
        )
        print("✅ Conexión exitosa a PostgreSQL")

        # Crear cursor y ejecutar prueba
        cur = conn.cursor()
        cur.execute("SELECT NOW();")
        result = cur.fetchone()
        print("🕒 Hora actual en la base:", result[0])

        # Verificar si la tabla existe
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'muestras_imagenes'
            );
        """
        )
        table_exists = cur.fetchone()[0]
        print("📦 ¿Tabla 'muestras_imagenes' existe?:", table_exists)

        # Cerrar conexión
        cur.close()
        conn.close()
        print("🔌 Conexión cerrada")

    except Exception as e:
        print("❌ Error al conectar:", e)


if __name__ == "__main__":
    test_connection()
