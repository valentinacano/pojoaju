# Proyecto Pojoaju

Proyecto para captura, procesamiento y entrenamiento de lenguaje de señas utilizando MediaPipe y redes neuronales.

## Requisitos

- Python 3.10
- PostgreSQL
- HDF5
- Pipenv
- Sphinx (documentación)

## Setup del entorno

```bash
# Instalar Python y HDF5
brew install python@3.10
brew install hdf5

# PostgreSQL
brew install postgresql
brew services start postgresql

# Crear y activar entorno virtual
pipenv --python 3.10
pipenv install
pipenv shell

# Instalar el proyecto en modo editable
pipenv run pip install -e .
```
### ¿Por qué pip install -e .?
Este comando instala el proyecto en modo editable, lo que significa que cualquier cambio que hagas en el código fuente (app/, ml/, etc.) se aplica automáticamente sin necesidad de reinstalar.

- Solo necesitás ejecutarlo una vez, luego de clonar el proyecto o crear el entorno virtual.
- Si modificás setup.py o reinstalás el entorno (pipenv --rm + pipenv install), corrélo de nuevo.

Podés verificar si ya está instalado con:
```bash
pipenv run pip list
```
Y deberías ver:
```bash
pojoaju    0.1.0    editable
```

###  Control de versiones y CHANGELOG
Cada vez que el proyecto evoluciona, es recomendable actualizar el número de versión en setup.py.
```bash
version="0.1.0"
```
#### ¿Cuándo cambiar la versión?
| Situación                                                       | Ejemplo versión |
|----------------------------------------------------------------|------------------|
| Primer release mínimo funcional                                | 0.1.0            |
| Se agrega una funcionalidad visible o significativa            | 0.2.0            |
| Se hace una corrección o ajuste menor                          | 0.1.1            |
| Se rompe compatibilidad o cambia la estructura de uso general  | 1.0.0            |

#### ¿Qué es un changelog?
Un changelog es un archivo que documenta los cambios por versión. Se recomienda usar un archivo CHANGELOG.md con entradas como esta:
```bash
## [0.2.0] - 2025-07-01
### Agregado
- Soporte para traducción de colores y emociones
- Panel de vista previa en Flask

### Corregido
- Error en normalización de keypoints vacíos
```
Esto ayuda a saber qué cambió, cuándo y por qué.

## Estructura de carpetas propuesta

```plaintext
pojoaju/
├── Pipfile / Pipfile.lock      # Dependencias del entorno
├── README.md                   # Este archivo
├── main.py / run.py            # Entradas principales
├── app/                        # Configuración, base de datos, vistas
│   ├── config.py
│   ├── database/               # Conexión y queries SQL
│   ├── models/
│   ├── services/
│   └── views/                  # Flask GUI
├── ml/                         # Módulos de machine learning
│   ├── features/               # Ingeniería de features (keypoints)
│   ├── models/                 # Modelos entrenados
│   ├── prediction/             # Scripts para predicción
│   ├── training/               # Entrenamiento del modelo
│   └── utils/                  # Funciones auxiliares
├── data/
│   ├── frame_actions/          # Frames capturados por palabra
│   └── models/                 # Modelos `.keras` guardados
├── docs/                       # Documentación generada por Sphinx
├── static/                     # Recursos estáticos
├── tests/                      # Pruebas unitarias
└── build_docs.py               # Ejecución de la documentación
```

| Carpeta              | Descripción                                                                 |
|----------------------|------------------------------------------------------------------------------|
| `app/`               | Módulo principal de la aplicación. Contiene configuración, vistas y acceso a datos. |
| `app/config.py`      | Parámetros globales de configuración (paths, conexión a BD, etc.).          |
| `app/database/`      | Lógica de conexión, creación y consultas a la base de datos PostgreSQL.     |
| `app/models/`        | (Reservado) Clases y estructuras de datos, si se definen entidades.         |
| `app/services/`      | (Reservado) Lógica de negocio y orquestación de funcionalidades.            |
| `app/utils/`         | Funciones auxiliares y herramientas reutilizables para toda la app.         |
| `app/views/`         | Vistas de la app, incluyendo interfaz Flask (`flask_gui.py`) y plantillas. |
| `ml/`                | Módulo de machine learning del proyecto.                                    |
| `ml/features/`       | Scripts de ingeniería de features: captura, normalización y procesamiento.  |
| `ml/training/`       | Scripts de entrenamiento y definición de modelo.                            |
| `ml/prediction/`     | (Reservado) Scripts para realizar predicciones a futuro.                    |
| `ml/models/`         | (Reservado) Almacenamiento de modelos exportados (ej: `.keras`).            |
| `ml/utils/`          | Funciones específicas para ML (keypoints, normalización, entrenamiento).    |
| `data/`              | Carpeta de datos usados por el sistema.                                     |
| `data/frame_actions/`| Muestras de video procesadas por palabra, organizadas por carpeta.          |
| `data/models/`       | Modelos entrenados en formato `.keras`.                                     |
| `docs/`              | Documentación generada con Sphinx. Contiene `.rst`, `.html`, etc.           |
| `static/`            | Archivos estáticos como CSS y HTML para la interfaz.                        |
| `tests/`             | Pruebas unitarias y de integración para validar funciones del proyecto.     |
| `main.py` / `run.py` | Puntos de entrada del sistema para ejecutar procesos o levantar la app.     |
| `build_docs.py`      | Script para generar automáticamente la documentación con Sphinx.            |
| `Pipfile`            | Definición de dependencias del entorno virtual (pipenv).                    |
| `Pipfile.lock`       | Versión exacta de dependencias instaladas (lockfile).                       |
| `README.md`          | Documentación general del proyecto.                                         |

## Documentación

La documentación está generada con Sphinx y se encuentra en el directorio docs/.

#### Para construir la documentación localmente:
```bash
pip install sphinx
sphinx-apidoc -o docs/source app ml
make -C docs html
open docs/build/html/index.html
```
#### O directamente:
```bash
python build_docs.py
```

## Testing
Las pruebas están en la carpeta tests/.

#### Para ejecutar los tests:
```bash
pipenv install --dev pytest
pipenv run pytest
```

## Traducción Inicial

### A - Z & 0 - 9
Incluye todas las letras mayúsculas y minúsculas (A-Z, a-z) y números del 0 al 9.

### Meses y días
Incluye todos los nombres de los meses del año y días de la semana.

### Palabras por categoría

| Categoría                     | Palabra                                                                           |
|-------------------------------|-----------------------------------------------------------------------------------|
| Animales                      | Perro, Gato, Vaca, Caballo, Cerdo, Gallina, Pájaro, Ratón                         |
| Básicos                       | Desayuno, Almuerzo, Cena, Baño, Comer, Tomar, Dormir, Sueño, Hambre, Sed          |
| Colores                       | Rojo, Azul, Verde, Amarillo, Negro, Blanco, Naranja                               |
| Emociones                     | Feliz, Triste, Enojado/a, Asustado/a, Cansado/a, Llorar, Reír, Amar               |
| Familia y personas            | Mamá, Papá, Hermano, Hermana, Abuela, Abuelo, Mujer, Hombre                       |
| Saludos y expresiones básicas | Hola, Chau, Buenos días, Buenas tardes, Buenas noches, Gracias, Por favor, Perdón |
| Tiempo                        | Hoy, Ayer, Mañana, Tarde, Noche, Hora, Minuto                                     |
