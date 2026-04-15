# Pojoaju

Sistema de reconocimiento de lengua de señas paraguaya (LSPy) usando MediaPipe y redes neuronales LSTM.

---

## Requisitos previos

- **Python 3.10** — versión exacta requerida
- **PostgreSQL** — base de datos del proyecto
- **pipenv** — gestor de entorno virtual (opcional, también funciona con pip)

### Instalación en macOS

```bash
brew install python@3.10
brew install postgresql
brew services start postgresql
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

---

## Setup desde cero

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd pojoaju
```

### 2. Crear las bases de datos

```bash
psql postgres -c "CREATE DATABASE pojoaju;"
psql postgres -c "CREATE DATABASE pojoaju_test;"
```

En macOS con Homebrew el usuario por defecto es tu usuario del sistema:

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

Completá `.env` con tus credenciales:

```
DB_USER=tu_usuario_postgresql
DB_PASSWORD=
DB_HOST=localhost
DB_PORT=5432
FLASK_SECRET=cualquier-string-secreto
```

### 4. Instalar dependencias

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
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 5. Crear carpetas necesarias

```bash
mkdir -p data/frames data/models data/exports static/confusion scripts
```

### 6. Crear archivos `__init__.py`

```bash
touch app/__init__.py app/database/__init__.py app/services/__init__.py
touch app/views/__init__.py ml/__init__.py tests/__init__.py
```

### 7. Levantar el servidor

```bash
python main.py
```

Abrí el navegador en: **http://127.0.0.1:5000**

---

## Estructura del proyecto

```
pojoaju/
├── main.py                        # Entry point
├── run.py                         # Formateo + ejecución
├── setup.py
├── requirements.txt
├── Pipfile
├── pytest.ini
├── .env.example
├── .gitignore
├── build_docs.py                  # Genera documentación Sphinx
│
├── app/
│   ├── config.py                  # Paths, constantes del modelo, config de BD
│   ├── database/
│   │   ├── connection.py
│   │   ├── schema.py
│   │   └── queries.py             # Todo el SQL del proyecto
│   ├── services/
│   │   └── text_to_speech.py
│   └── views/
│       └── flask_gui.py
│
├── ml/
│   ├── capture.py                 # Captura desde cámara y video
│   ├── normalize.py               # Normalización (método único)
│   ├── keypoints.py               # Extracción de 1662 features con MediaPipe
│   ├── model.py                   # Arquitectura LSTM
│   ├── train.py                   # Pipeline de entrenamiento
│   ├── predict.py                 # Predicción en tiempo real
│   ├── evaluate.py                # Matriz de confusión
│   ├── sign_animator.py           # Animación de señas para texto/voz a señas
│   └── pipeline.py                # Orquestador general
│
├── scripts/                       # Herramientas de mantenimiento
│   ├── check_outliers.py          # Analiza calidad de muestras
│   └── clean_outliers.py          # Elimina muestras atípicas
│
├── templates/                     # Templates HTML de Flask
├── static/
│   ├── css/styles.css
│   └── img/
│
├── tests/
│   ├── conftest.py
│   ├── test_normalize.py
│   ├── test_database.py
│   └── test_keypoints.py
│
├── docs/                          # Documentación Sphinx
│   └── source/
│
└── data/                          # Local, no se sube al repo
    ├── frames/
    ├── models/
    └── exports/
```

---

## Flujo de uso completo

### Paso 1 — Capturar muestras

1. Ir a **Diccionario** → **Tomar Muestras**
2. Elegir **Grabar Video** o **Subir Video**
3. Repetir hasta tener al menos **80 muestras por palabra**

### Paso 2 — Entrenar el modelo

1. Ir a **Entrenamiento → Entrenar Modelo**
2. El modelo se guarda en `data/models/`

El entrenamiento usa `EarlyStopping(patience=30)`.

### Paso 3 — Traducir señas en tiempo real

Ir a **Traducir Señas** — predice en tiempo real desde la cámara.

### Paso 4 — Texto a Señas

Ir a **Texto a Señas** — escribí una palabra y ve su seña animada.

### Paso 5 — Voz a Señas

Ir a **Voz a Señas** (requiere Chrome) — hablá y el sistema muestra la seña automáticamente.

### Paso 6 — Evaluar el modelo

Ir a **Matriz de Confusión** para ver qué palabras se confunden.

---

## Testing

```bash
# Todos los tests
TESTING=1 python -m pytest -v

# Test específico
TESTING=1 python -m pytest tests/test_normalize.py -v

# Windows
set TESTING=1 && python -m pytest -v
```

---

## Documentación

```bash
python build_docs.py
```

O manualmente:

```bash
cd docs
make html
open build/html/index.html
```

---

## Herramientas de mantenimiento

Estas herramientas se usan manualmente para auditar y mejorar la calidad de los datos. No son parte del flujo automático.

### Analizar outliers

Muestra las muestras que están muy alejadas del centroide de su palabra, sin eliminar nada:

```bash
python scripts/check_outliers.py
```

### Limpiar outliers

Primero corré en modo simulación para ver qué va a eliminar:

```bash
python scripts/clean_outliers.py
```

Cuando estés conforme, abrí el archivo y cambiá `dry_run=True` a `dry_run=False` y volvé a ejecutar. Las muestras eliminadas se borran de la BD permanentemente.

**Cuándo usar esto:**
- Cuando el val_accuracy es muy inestable entre entrenamientos
- Cuando sospechás que algunas capturas fueron incorrectas
- Después de agregar muchas muestras nuevas

---

## Decisiones de diseño importantes

**`normalize_sequence()` es el método único del proyecto.** Se usa en `train.py`, `predict.py` y `evaluate.py`. Usar métodos distintos entre training y predicción baja drásticamente la accuracy.

**`ml/` es independiente de Flask.** Toda la comunicación pasa por `ml/pipeline.py`.

**Una sola implementación de TTS.** `app/services/text_to_speech.py` es la única fuente.

---

## Solución de problemas comunes

**Puerto 5000 ocupado (macOS)**
Ejecutar `app.run(debug=False)` sin especificar host ni port.

**Error de conexión a PostgreSQL**
Verificar credenciales en `.env` y que PostgreSQL esté corriendo.

**Cámara no detectada**
Cambiar `camera_index` en `app/config.py` de `0` a `1` o `2`.

**Val accuracy inestable**
Capturar más muestras (mínimo 80 por palabra) y correr `scripts/check_outliers.py` para detectar muestras problemáticas.

**Voz a Señas no funciona**
Usar Chrome — la Web Speech API no está disponible en Safari ni Firefox.

---

## Versión

Ver `CHANGELOG.md` para el historial completo de cambios.