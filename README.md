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
GEMINI_API_KEY=tu_api_key_de_gemini
```

### 4. Obtener la API key de Gemini

La función de **traducción de frases** (LSPy → español natural) usa la API de Google Gemini.

1. Ir a **https://aistudio.google.com**
2. Iniciar sesión con una cuenta de Google
3. Hacer clic en **Get API Key** → **Create API Key**
4. Copiar la key y pegarla en `.env` como `GEMINI_API_KEY`

> **Importante:** el modelo usado es `gemini-2.5-flash-lite`. Verificá que tu cuenta tenga acceso a este modelo ejecutando:
> ```bash
> python -c "
> from dotenv import load_dotenv; load_dotenv()
> import os, google.generativeai as genai
> genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
> model = genai.GenerativeModel('gemini-2.5-flash-lite')
> print(model.generate_content('OK').text)
> "
> ```
> Si recibís un error de quota, intentá con una cuenta de Google de otro país (Argentina, EEUU) ya que el free tier no está disponible en todas las regiones.

> **Sin API key:** el sistema igual funciona — simplemente muestra las señas detectadas sin traducir al español natural.

### 5. Instalar dependencias

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

### 6. Crear carpetas necesarias

```bash
mkdir -p data/frames data/models data/exports static/confusion scripts
```

### 7. Crear archivos `__init__.py`

```bash
touch app/__init__.py app/database/__init__.py app/services/__init__.py
touch app/views/__init__.py ml/__init__.py tests/__init__.py
```

### 8. Levantar el servidor

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
│   ├── predict.py                 # Predicción en tiempo real + buffer de frases
│   ├── evaluate.py                # Matriz de confusión
│   ├── sign_animator.py           # Animación de señas para texto/voz a señas
│   ├── translator.py              # Traducción LSPy → español con Gemini
│   └── pipeline.py                # Orquestador general
│
├── scripts/                       # Herramientas de mantenimiento
│   ├── check_outliers.py          # Analiza calidad de muestras
│   ├── clean_outliers.py          # Elimina muestras atípicas
│   ├── augment_data.py            # Genera muestras sintéticas
│   ├── export_data.py             # Exporta datos para compartir
│   └── import_data.py             # Importa datos sin conflictos de IDs
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
3. Repetir hasta tener al menos **80 muestras reales por palabra**

### Paso 2 — Entrenar el modelo

1. Ir a **Entrenamiento → Entrenar Modelo**
2. El modelo se guarda en `data/models/`

El entrenamiento usa `EarlyStopping(patience=30)`.

### Paso 3 — Traducir señas en tiempo real

1. Ir a **Traducir Señas**
2. Hacé tus señas frente a la cámara
3. Las palabras detectadas se acumulan en **Señas detectadas**
4. Hacé clic en **✨ Traducir frase** para convertir la secuencia al español natural con Gemini

### Paso 4 — Texto a Señas

Ir a **Texto a Señas** — escribí una palabra y ve su seña animada con el stickman.

### Paso 5 — Voz a Señas

Ir a **Voz a Señas** (requiere Chrome) — hablá y el sistema muestra la seña automáticamente.

### Paso 6 — Evaluar el modelo

Ir a **Matriz de Confusión** para ver qué palabras se confunden entre sí.

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

Estas herramientas se usan manualmente para auditar y mejorar la calidad de los datos.

### Analizar outliers

```bash
python scripts/check_outliers.py
```

### Limpiar outliers

```bash
# Primero simulación
python scripts/clean_outliers.py

# Cuando estés conforme, cambiá dry_run=True → dry_run=False y volvé a ejecutar
```

### Generar muestras sintéticas

```bash
# Primero simulación
python scripts/augment_data.py

# Cambiá dry_run=True → dry_run=False para generar de verdad
```

### Exportar e importar datos entre compañeros

```bash
# Exportar (genera pojoaju_export.json)
python scripts/export_data.py

# Importar (primero simulación, luego dry_run=False)
python scripts/import_data.py pojoaju_export.json
```

---

## Decisiones de diseño importantes

**`normalize_sequence()` es el método único del proyecto.** Se usa en `train.py`, `predict.py` y `evaluate.py`. Usar métodos distintos entre training y predicción baja drásticamente la accuracy.

**`ml/` es independiente de Flask.** Toda la comunicación pasa por `ml/pipeline.py`.

**Una sola implementación de TTS.** `app/services/text_to_speech.py` es la única fuente.

**Traducción con Gemini es opcional.** Si no hay `GEMINI_API_KEY` configurada, el sistema muestra las señas detectadas sin traducir. El resto del sistema funciona normalmente.

---

## Solución de problemas comunes

**Puerto 5000 ocupado (macOS)**
Ejecutar `app.run(debug=False)` sin especificar host ni port.

**Error de conexión a PostgreSQL**
Verificar credenciales en `.env` y que PostgreSQL esté corriendo.

**Cámara no detectada**
Cambiar `camera_index` en `app/config.py` de `0` a `1` o `2`.

**Val accuracy inestable**
Capturar más muestras reales (mínimo 80 por palabra) y correr `scripts/check_outliers.py`.

**Voz a Señas no funciona**
Usar Chrome — la Web Speech API no está disponible en Safari ni Firefox.

**Error de quota en Gemini (`limit: 0`)**
El free tier no está disponible en todas las regiones. Intentá con una cuenta de Google de otro país o verificá tu plan en https://ai.dev/rate-limit.

**Traducción muestra las señas sin traducir**
Verificar que `GEMINI_API_KEY` esté correctamente configurada en `.env` y que el modelo `gemini-2.5-flash-lite` sea accesible desde tu cuenta.

---

## Versión

Ver `CHANGELOG.md` para el historial completo de cambios.
