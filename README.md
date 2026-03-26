# Pojoaju

Sistema de reconocimiento de lengua de señas paraguaya (LSPy) usando MediaPipe y redes neuronales LSTM.

---

## Requisitos previos

Antes de arrancar, necesitás tener instalado:

- **Python 3.10** — versión exacta requerida
- **PostgreSQL** — base de datos del proyecto
- **pipenv** — gestor de entorno virtual (opcional, también funciona con pip)

### Instalación en macOS

```bash
brew install python@3.10
brew install postgresql
brew services start postgresql   # inicia PostgreSQL como servicio
```

### Instalación en Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
sudo apt install postgresql postgresql-contrib
sudo service postgresql start
```

### Instalación en Windows

1. Descargar Python 3.10 desde https://python.org
2. Descargar PostgreSQL desde https://www.postgresql.org/download/windows/
3. Durante la instalación de PostgreSQL, recordar el usuario (`postgres`) y la contraseña que se configura.

---

## Setup desde cero

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd pojoaju
```

### 2. Crear las bases de datos en PostgreSQL

```bash
# Conectarse a PostgreSQL
psql -U postgres

# Dentro de psql, ejecutar:
CREATE DATABASE pojoaju;
CREATE DATABASE pojoaju_test;
\q
```

En macOS con Homebrew el usuario por defecto puede ser tu usuario del sistema:

```bash
psql postgres
CREATE DATABASE pojoaju;
CREATE DATABASE pojoaju_test;
\q
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

Abrí `.env` y completá con tus credenciales:

```
DB_USER=postgres          # o tu usuario de PostgreSQL
DB_PASSWORD=              # la contraseña que configuraste
DB_HOST=localhost
DB_PORT=5432
FLASK_SECRET=cualquier-string-secreto
```

### 4. Crear el entorno virtual e instalar dependencias

**Con pipenv (recomendado):**

```bash
pip install pipenv
pipenv --python 3.10
pipenv install
pipenv shell
pipenv run pip install -e .
```

**Con pip:**

```bash
python3.10 -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Para verificar que el proyecto quedó instalado en modo editable:

```bash
pip list | grep pojoaju
# Debería mostrar: pojoaju   1.0.0   editable
```

### 5. Crear carpetas necesarias

```bash
mkdir -p data/frames data/models data/exports static/confusion
```

### 6. Crear archivos `__init__.py`

```bash
touch app/__init__.py
touch app/database/__init__.py
touch app/services/__init__.py
touch app/views/__init__.py
touch ml/__init__.py
touch tests/__init__.py
```

En Windows (PowerShell):

```powershell
New-Item app/__init__.py, app/database/__init__.py, app/services/__init__.py, app/views/__init__.py, ml/__init__.py, tests/__init__.py -ItemType File
```

### 7. Levantar el servidor

```bash
python main.py
```

Abrí el navegador en: **http://localhost:5000**

---

## Estructura del proyecto

```
pojoaju/
├── main.py                        # entry point — inicializa BD y levanta Flask
├── run.py                         # formatea con black + ejecuta main.py
├── setup.py                       # instalación editable del proyecto
├── requirements.txt               # dependencias para pip
├── Pipfile                        # dependencias para pipenv
├── pytest.ini                     # configuración de tests
├── .env.example                   # plantilla de variables de entorno
├── .gitignore
│
├── app/
│   ├── config.py                  # paths, constantes del modelo, config de BD
│   ├── database/
│   │   ├── connection.py          # conexión PostgreSQL
│   │   ├── schema.py              # creación de tablas
│   │   └── queries.py             # todo el SQL del proyecto
│   ├── services/
│   │   └── text_to_speech.py      # texto a voz (única implementación)
│   └── views/
│       └── flask_gui.py           # rutas Flask
│
├── ml/
│   ├── capture.py                 # captura desde cámara y video pregrabado
│   ├── normalize.py               # normalización de secuencias (método único)
│   ├── keypoints.py               # extracción de 1662 features con MediaPipe
│   ├── model.py                   # arquitectura LSTM
│   ├── train.py                   # pipeline de entrenamiento
│   ├── predict.py                 # predicción en tiempo real
│   ├── evaluate.py                # matriz de confusión y métricas
│   └── pipeline.py                # orquestador general (usado por Flask)
│
├── templates/                     # templates HTML de Flask
├── static/
│   ├── css/styles.css             # estilos globales
│   └── img/                       # íconos e imágenes
│
├── tests/
│   ├── conftest.py                # fixtures de pytest
│   ├── test_normalize.py
│   ├── test_database.py
│   └── test_keypoints.py
│
└── data/                          # generado localmente, NO se sube al repo
    ├── frames/                    # muestras capturadas por palabra
    ├── models/                    # modelos .keras entrenados
    └── exports/                   # videos subidos para entrenamiento
```

---

## Flujo de uso completo

### Paso 1 — Capturar muestras para una palabra

1. Ir a **Diccionario** desde la pantalla principal
2. Buscar la palabra que querés entrenar
3. Hacer clic en **Tomar Muestras**
4. Elegir **Grabar Video** (cámara en vivo) o **Subir Video** (video pregrabado)
5. Al finalizar, el sistema normaliza los frames y guarda los keypoints en la BD

Repetir para cada palabra. Se recomiendan **al menos 30 muestras por palabra** para buena accuracy.

### Paso 2 — Entrenar el modelo

1. Ir a **Entrenamiento → Entrenar Modelo**
2. Hacer clic en **Iniciar Entrenamiento**
3. Esperar — puede tomar varios minutos dependiendo de la cantidad de muestras
4. El modelo se guarda automáticamente en `data/models/`

El entrenamiento usa `EarlyStopping(patience=30)` — se detiene solo cuando deja de mejorar.

### Paso 3 — Traducir señas en tiempo real

1. Ir a **Traducir Señas** desde la pantalla principal
2. El sistema abre la cámara y empieza a predecir en tiempo real
3. Las predicciones aparecen en pantalla con su nivel de confianza

### Paso 4 — Evaluar el modelo (opcional)

1. Ir a **Matriz de Confusión**
2. Hacer clic en **Generar Matriz**
3. Ver qué palabras se confunden entre sí para decidir qué muestras reforzar

---

## Testing

```bash
# Correr todos los tests (usa pojoaju_test automáticamente)
TESTING=1 pytest -v

# Test específico
TESTING=1 pytest tests/test_normalize.py -v

# En Windows
set TESTING=1 && pytest -v
```

---

## Decisiones de diseño importantes

**`normalize_sequence()` es el método único del proyecto.** Se usa en `train.py`, `predict.py` y `evaluate.py`. Usar métodos distintos entre training y predicción es el error más común que baja la accuracy.

**`ml/` es independiente de Flask.** Los módulos de ML no importan nada de Flask. Toda la comunicación pasa por `ml/pipeline.py`.

**Una sola implementación de TTS.** `app/services/text_to_speech.py` es la única fuente.

---

## Solución de problemas comunes

**Error de conexión a PostgreSQL:**
```
❌ Error de conexión: FATAL: password authentication failed
```
Verificar que las credenciales en `.env` sean correctas y que PostgreSQL esté corriendo.

**ModuleNotFoundError al importar mediapipe u otro paquete:**
Asegurarse de estar dentro del entorno virtual (`pipenv shell` o `source venv/bin/activate`) y que `pip install -e .` fue ejecutado.

**Cámara no detectada:**
Verificar que `camera_index=0` en `app/config.py` sea correcto. Cambiar a `1` o `2` si hay múltiples cámaras conectadas.

**El modelo predice siempre la misma palabra:**
Verificar que hay al menos 2 palabras con keypoints antes de entrenar. Correr `python test.py` en la raíz para ver el conteo de muestras por palabra.

**`pip install -e .` falla:**
Asegurarse de que `setup.py` está en la raíz y que el entorno virtual está activo.

---

## Versión

Ver `CHANGELOG.md` para el historial completo de cambios.