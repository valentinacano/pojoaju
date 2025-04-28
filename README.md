# Proyecto Pojoaju

Este proyecto requiere **Python 3.10** y algunas dependencias específicas. A continuación se detallan los pasos para configurar el entorno y la estructura del proyecto.

## Setup del entorno

### Instalación de herramientas bases

#### Python y gestor de paquetes:
```bash
brew install python@3.10
```
#### Librería para archivos HDF5
```bash
brew install hdf5
```
#### Base de datos PostgreSQL
```bash
brew install postgresql
brew services start postgresql (activacion de la base de datos)
```
### Crear y activar entorno virtual
#### Crear entorno virtual usando Python 3.10
```bash
pipenv --python 3.10
```
#### Instalar dependencias en el entorno
```bash
pipenv install
```
#### Activar entorno
```bash
pipenv shell
```
## Documentación

```bash
pip install sphinx
sphinx-quickstart docs
sphinx-apidoc -o docs/source app ml
make html
open docs/build/html/index.html
# o ejecutar
python build_docs.py
```


## Estructura de carpetas propuesta

```plaintext
pojoaju/
├── app/                   
│   ├── models/             # Clases y lógica de datos (entidades, estructuras)
│   ├── views/              # Presentación (consola, HTML, UI)
│   ├── services/           # Lógica de negocio y procesamiento
│   ├── utils/              # Funciones auxiliares o genéricas
│   ├── main.py             # Punto de entrada
│   └── config.py           # Configuraciones generales (paths, parámetros)
├── ml/                     
│   ├── training/           # Scripts de entrenamiento
│   ├── prediction/         # Scripts de predicción
│   ├── features/           # Ingeniería de features
│   └── models/             # Modelos ya entrenados (.pkl, .joblib, etc.)
├── notebooks/              # Jupyter Notebooks para exploración y pruebas
├── data/                   
│   ├── raw/                # Datos crudos
│   ├── processed/          # Datos procesados / limpios
│   └── external/           # Datos externos (de terceros)
├── db/                     # Conexión, esquemas y lógica de acceso a BD
├── tests/                  # Pruebas del proyecto
├── static/                 
│   ├── html/
│   └── css/
├── requirements.txt        # Dependencias
├── instalacion.sh          # Script de instalación automática
└── README.md               # Instrucciones del proyecto
```

| Carpeta             | Descripción                                                                 |
|---------------------|------------------------------------------------------------------------------|
| `app/`              | Código base de la aplicación. Contiene la lógica general y configuración.    |
| `app/models/`       | Clases y estructuras de datos (por ejemplo, clases tipo `User`, `Record`).   |
| `app/views/`        | Visualización o presentación (CLI, HTML, dashboards, si aplica).             |
| `app/services/`     | Lógica de negocio: procesos, validaciones, transformaciones generales.       |
| `app/utils/`        | Funciones auxiliares reutilizables (logs, fechas, formateos, etc.).          |
| `app/main.py`       | Punto de entrada del proyecto. Junta todo lo necesario para ejecutar.        |
| `app/config.py`     | Parámetros globales del proyecto (paths, flags, hiperparámetros, etc.).      |
| `ml/`               | Código específico de machine learning.                                       |
| `ml/training/`      | Scripts para entrenamiento de modelos.                                       |
| `ml/prediction/`    | Scripts para hacer predicciones con modelos entrenados.                      |
| `ml/features/`      | Scripts de ingeniería de features: transformación y selección de variables.  |
| `ml/models/`        | Modelos entrenados guardados (.pkl, .joblib, etc.).                          |
| `notebooks/`        | Notebooks Jupyter para análisis exploratorio y pruebas.                      |
| `data/`             | Datos utilizados en el proyecto.                                             |
| `data/raw/`         | Datos originales sin procesar.                                               |
| `data/processed/`   | Datos limpios, transformados y listos para entrenar.                         |
| `data/external/`    | Datos externos de terceros.                                                  |
| `db/`               | Lógica de conexión, consultas y definiciones de base de datos.               |
| `tests/`            | Pruebas unitarias y de integración para validar el código.                   |
| `static/`           | Recursos estáticos como HTML, CSS, JS (si hay una interfaz visual).          |
| `requirements.txt`  | Lista de dependencias del proyecto.                                          |
| `instalacion.sh`    | Script para automatizar instalación de entorno y dependencias.              |
| `README.md`         | Documentación general del proyecto. 


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
