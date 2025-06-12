import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 15

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "data/frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "data/models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
DB_CONFIG = {
    "dbname": "pojoaju",
    "user": "valentinacano",
    "password": "",  # o ponerlo en un .env si preferís más seguridad
    "host": "localhost",
    "port": "5432",
}


# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# PALABRAS & CATEGORIAS INICIALES A SER ENTRENADAS

categories = [
    "Animales",
    "Básicos",
    "Colores",
    "Emociones",
    "Familia y personas",
    "Saludos y expresiones básicas",
    "Tiempo",
]

words = {
    "Animales": [
        "Perro",
        "Gato",
        "Vaca",
        "Caballo",
        "Cerdo",
        "Gallina",
        "Pájaro",
        "Ratón",
    ],
    "Básicos": [
        "Desayuno",
        "Almuerzo",
        "Cena",
        "Baño",
        "Comer",
        "Tomar",
        "Dormir",
        "Sueño",
        "Hambre",
        "Sed",
    ],
    "Colores": ["Rojo", "Azul", "Verde", "Amarillo", "Negro", "Blanco", "Naranja"],
    "Emociones": [
        "Feliz",
        "Triste",
        "Enojado/a",
        "Asustado/a",
        "Cansado/a",
        "Llorar",
        "Reír",
        "Amar",
    ],
    "Familia y personas": [
        "Mamá",
        "Papá",
        "Hermano",
        "Hermana",
        "Abuela",
        "Abuelo",
        "Mujer",
        "Hombre",
    ],
    "Saludos y expresiones básicas": [
        "Hola",
        "Chau",
        "Buenos días",
        "Buenas tardes",
        "Buenas noches",
        "Gracias",
        "Por favor",
        "Perdón",
    ],
    "Tiempo": ["Hoy", "Ayer", "Mañana", "Tarde", "Noche", "Hora", "Minuto"],
}
