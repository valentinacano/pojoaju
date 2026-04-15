"""
Configuración global del proyecto Pojoaju.

Centraliza todos los parámetros del sistema: rutas, base de datos,
constantes del modelo y configuración de OpenCV.
"""

import os
import sys
import cv2

# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------

# Cantidad de frames por secuencia (debe ser igual en captura, training y predicción)
MODEL_FRAMES = 15

# Cantidad de features por frame extraídos por MediaPipe Holistic
# pose: 33*4=132, face: 468*3=1404, left_hand: 21*3=63, right_hand: 21*3=63
LENGTH_KEYPOINTS = 1662

# Umbral mínimo de confianza para aceptar una predicción
PREDICTION_THRESHOLD = 0.7

# Frames mínimos capturados para considerar una seña válida
MIN_FRAMES_CAPTURED = 10

# Frames de cooldown entre predicciones (evita repeticiones)
PREDICTION_COOLDOWN = 20

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------

ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH, "data")
FRAMES_PATH = os.path.join(DATA_PATH, "frames")        # muestras capturadas
MODELS_PATH = os.path.join(DATA_PATH, "models")        # modelos entrenados
EXPORTS_PATH = os.path.join(DATA_PATH, "exports")      # videos subidos

MODEL_PATH = os.path.join(MODELS_PATH, f"pojoaju_{MODEL_FRAMES}.keras")

# ---------------------------------------------------------------------------
# BASE DE DATOS
# ---------------------------------------------------------------------------

DB_PROD = {
    "dbname": "pojoaju",
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

DB_TEST = {
    "dbname": "pojoaju_test",
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

def get_db_config():
    """
    Retorna la configuración de base de datos correcta según el entorno.

    Detecta automáticamente si estamos en un entorno de testing (pytest)
    y devuelve la config correspondiente.

    Returns:
        dict: Configuración de conexión para psycopg2.
    """
    is_testing = (
        os.getenv("TESTING") == "1"
        or "PYTEST_CURRENT_TEST" in os.environ
        or any("pytest" in arg for arg in sys.argv)
    )
    return DB_TEST if is_testing else DB_PROD

DB_CONFIG = get_db_config()

# ---------------------------------------------------------------------------
# CAPTURA
# ---------------------------------------------------------------------------

# Frames a descartar al inicio y fin de cada muestra
MARGIN_FRAMES = 1

# Mínimo de frames válidos para guardar una muestra
MIN_FRAMES_SAMPLE = 5

# Frames de espera antes de cortar la grabación al perder la mano
DELAY_FRAMES = 3

# ---------------------------------------------------------------------------
# DISPLAY (OpenCV)
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# ---------------------------------------------------------------------------
# VOCABULARIO INICIAL
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Animales",
    "Básicos",
    "Colores",
    "Emociones",
    "Familia y personas",
    "Saludos y expresiones básicas",
    "Tiempo",
]

WORDS = {
    "Animales": ["Perro", "Gato", "Vaca", "Caballo", "Cerdo", "Gallina", "Pájaro", "Ratón"],
    "Básicos": ["Desayuno", "Almuerzo", "Cena", "Baño", "Comer", "Tomar", "Dormir", "Sueño", "Hambre", "Sed"],
    "Colores": ["Rojo", "Azul", "Verde", "Amarillo", "Negro", "Blanco", "Naranja"],
    "Emociones": ["Feliz", "Triste", "Enojado", "Asustado", "Cansado", "Llorar", "Reír", "Amar"],
    "Familia y personas": ["Mamá", "Papá", "Hermano", "Hermana", "Abuela", "Abuelo", "Mujer", "Hombre"],
    "Saludos y expresiones básicas": ["Hola", "Chau", "Buenos días", "Buenas tardes", "Buenas noches", "Gracias", "Por favor", "Perdón"],
    "Tiempo": ["Hoy", "Ayer", "Mañana", "Tarde", "Noche", "Hora", "Minuto"],
}