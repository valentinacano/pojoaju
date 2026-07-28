"""
Interfaz web Flask — Pojoaju.

Rutas organizadas por sección:
- General:      /  /translate
- Diccionario:  /dictionary  /dictionary/search  /dictionary/insert
- Entrenamiento:/training  /capture  /video  /save  /train
- Predicción:   /video_feed_prediction
- Evaluación:   /confusion
- Texto/Voz:    /text_to_sign  /voice_to_sign
- Frases:       /api/translate_phrase  /api/clear_phrase  /api/current_phrase
"""

import os
from datetime import datetime

from flask import (
    Flask,
    render_template,
    Response,
    redirect,
    url_for,
    request,
    jsonify,
    flash,
)
from werkzeug.utils import secure_filename

from app.config import FRAMES_PATH, EXPORTS_PATH
from app.database.queries import (
    fetch_all_words,
    fetch_all_categories,
    insert_word,
    count_samples_per_word,
    get_word_by_name,
    word_to_id,
)
from ml.pipeline import (
    start_capture_camera,
    start_capture_video,
    stop_capture_camera,
    process_and_save,
    run_training,
    run_predict_stream,
    run_evaluation,
)
from ml.sign_animator import get_sign_animation, get_available_words
from ml.predict import get_current_phrase, clear_phrase
from ml.translator import translate_signs_to_spanish, spanish_to_lspy_sequence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates"),
)
print(BASE_DIR)
print(app.static_folder)
app.secret_key = os.getenv("FLASK_SECRET", "pojoaju-dev-secret")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MIN_VIDEO_SAMPLES = 1
MAX_VIDEO_SAMPLES = 200


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate")
def translate_page():
    return render_template("translate.html")


@app.route("/video_feed_prediction")
def video_feed_prediction():
    return Response(
        run_predict_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# Diccionario
# ---------------------------------------------------------------------------


@app.route("/dictionary")
def dictionary():
    words = fetch_all_words()
    word_ids = [w[0] for w in words]
    samples_count = count_samples_per_word(word_ids)

    total = len(words)
    con_muestras = sum(1 for wid in word_ids if samples_count.get(wid, 0) > 0)

    return render_template(
        "dictionary.html",
        words=words,
        samples_count=samples_count,
        total=total,
        con_muestras=con_muestras,
        sin_muestras=total - con_muestras,
    )


@app.route("/dictionary/search", methods=["POST"])
def dictionary_search():
    query = request.form.get("query", "").strip().lower()
    words = fetch_all_words()
    filtered = [w for w in words if query in w[1].lower() or query in w[2].lower()]
    return render_template("dictionary.html", words=filtered)


@app.route("/dictionary/insert", methods=["GET", "POST"])
def insert_word_form():
    if request.method == "POST":
        word = request.form.get("word", "").strip()
        category = (
            request.form.get("category_new", "").strip()
            or request.form.get("category_existing", "").strip()
        )

        if not word:
            flash("La palabra es obligatoria.", "error")
            return redirect(url_for("insert_word_form"))

        if not category:
            flash("Seleccioná o ingresá una categoría.", "error")
            return redirect(url_for("insert_word_form"))

        insert_word(word, category)
        return render_template("insert_success.html", word=word, category=category)

    categories = fetch_all_categories()
    return render_template("insert_word_form.html", categories=categories)


# ---------------------------------------------------------------------------
# Entrenamiento — captura
# ---------------------------------------------------------------------------


@app.route("/training")
def training():
    return render_template("training.html")


@app.route("/training/selector/<word_id>/<word>")
def training_selector(word_id, word):
    return render_template("training_selector.html", word_id=word_id, word=word)


@app.route("/training/capture/<word_id>/<word>")
def capture_page(word_id, word):
    return render_template("capture.html", word_id=word_id, word=word)


@app.route("/video_feed/<word>")
def video_feed(word):
    return Response(
        start_capture_camera(word), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop_capture", methods=["POST"])
def stop_capture_route():
    stop_capture_camera()
    word = request.form.get("word")
    word_id = request.form.get("word_id")
    if word and word_id:
        return redirect(url_for("save_samples", word=word, word_id=word_id))
    return redirect(url_for("training"))


@app.route("/training/upload/<word_id>/<word>", methods=["GET", "POST"])
def upload_video(word_id, word):
    if request.method == "GET":
        return render_template("upload_video.html", word_id=word_id, word=word)

    file = request.files.get("video_file")
    if not file:
        flash("No se recibió ningún archivo.", "error")
        return redirect(url_for("upload_video", word_id=word_id, word=word))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        flash(
            f"Formato no soportado. Usá: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}", "error"
        )
        return redirect(url_for("upload_video", word_id=word_id, word=word))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{secure_filename(word)}_{timestamp}{ext}"
    word_export_folder = os.path.join(EXPORTS_PATH, word.strip().lower())
    os.makedirs(word_export_folder, exist_ok=True)
    video_path = os.path.join(word_export_folder, filename)
    file.save(video_path)

    try:
        sample_count = int(request.form.get("sample_count", 1))
    except ValueError:
        sample_count = 1

    sample_count = max(MIN_VIDEO_SAMPLES, min(sample_count, MAX_VIDEO_SAMPLES))
    start_capture_video(word, video_path, sample_count)

    return redirect(url_for("save_samples", word=word, word_id=word_id))


@app.route("/save_samples/<word>/<word_id>")
def save_samples(word, word_id):
    process_and_save(word, word_id)
    return render_template("save_samples.html", word=word, word_id=word_id)


# ---------------------------------------------------------------------------
# Entrenamiento — modelo
# ---------------------------------------------------------------------------


@app.route("/train")
def train_page():
    return render_template("train_model.html")


@app.route("/train", methods=["POST"])
def train_model():
    try:
        results = run_training()
        if "error" in results:
            return jsonify(success=False, error=results["error"])
        return jsonify(success=True, results=results)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


@app.route("/confusion")
def confusion_page():
    return render_template("confusion_matrix.html")


@app.route("/api/confusion", methods=["POST"])
def confusion_api():
    try:
        data = run_evaluation()
        return jsonify(success=True, **data)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route("/api/confusion/status")
def confusion_status():
    image_path = os.path.join(app.static_folder, "confusion", "confusion_matrix.png")

    exists = os.path.isfile(image_path)
    last_generated = None

    if exists:
        timestamp = os.path.getmtime(image_path)
        last_generated = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    return jsonify(
        exists=exists,
        last_generated=last_generated,
        path=(
            url_for("static", filename="confusion/confusion_matrix.png")
            if exists
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Texto a Señas / Voz a Señas
# ---------------------------------------------------------------------------


@app.route("/text_to_sign")
def text_to_sign():
    words = get_available_words()
    return render_template("text_to_sign.html", available_words=words)


@app.route("/api/sign/<word>")
def get_sign(word):
    """Retorna la animación de keypoints para una palabra."""
    animation = get_sign_animation(word.strip().lower())
    if animation is None:
        return jsonify(success=False, error=f"No hay keypoints para '{word}'"), 404
    return jsonify(success=True, **animation)


@app.route("/voice_to_sign")
def voice_to_sign():
    words = get_available_words()
    return render_template("voice_to_sign.html", available_words=words)


# ---------------------------------------------------------------------------
# Traducción de frases LSPy → Español con Gemini
# ---------------------------------------------------------------------------


@app.route("/api/translate_phrase", methods=["POST"])
def translate_phrase():
    """Traduce la frase acumulada de señas al español natural usando Gemini."""
    signs = get_current_phrase()

    if not signs:
        return jsonify(success=False, error="No hay señas acumuladas.")

    translation = translate_signs_to_spanish(signs)
    clear_phrase()

    return jsonify(success=True, signs=signs, translation=translation)


@app.route("/api/clear_phrase", methods=["POST"])
def clear_phrase_route():
    """Limpia el buffer de señas acumuladas."""
    clear_phrase()
    return jsonify(success=True)


@app.route("/api/current_phrase")
def current_phrase():
    """Retorna las señas acumuladas actualmente."""
    signs = get_current_phrase()
    return jsonify(signs=signs)


@app.route("/api/last_prediction")
def last_prediction():
    """Retorna la última predicción realizada."""
    from ml.predict import get_last_prediction

    return jsonify(get_last_prediction())


@app.route("/api/phrase_to_signs", methods=["POST"])
def phrase_to_signs():
    print("🚨 ENTRÓ A PHRASE_TO_SIGNS")
    data = request.get_json()
    print(f"📥 Data recibida: {data}")
    """Convierte una frase en español a una secuencia de señas del diccionario."""
    phrase = data.get("phrase", "").strip() if data else ""
    print(f"📥 Frase recibida: {phrase}")

    if not phrase:
        return jsonify(success=False, error="No se recibió ninguna frase.")
    words = get_available_words()
    sequence = spanish_to_lspy_sequence(phrase, words)
    if not sequence:
        return jsonify(
            success=False, error="No se encontraron señas disponibles para esa frase."
        )
    print(f"📤 Secuencia generada: {sequence}")
    return jsonify(success=True, sequence=sequence)


@app.route("/api/sequence_animation", methods=["POST"])
def sequence_animation():
    """
    Recibe una lista de palabras y retorna todos sus frames concatenados
    en una sola animación continua con frames de transición entre palabras.
    """

    data = request.get_json()
    words = data.get("words", []) if data else []

    if not words:
        return jsonify(success=False, error="No se recibieron palabras.")

    all_frames = []
    pose_connections = None
    hand_connections = None
    word_markers = []
    TRANSITION_FRAMES = 8

    print(f"🔍 Palabras recibidas: {words}")
    for word in words:
        animation = get_sign_animation(word.strip().lower())
        print(f"  → '{word}' → {'OK' if animation else 'None'}")
        if animation is None:
            continue

        if pose_connections is None:
            pose_connections = animation["pose_connections"]
            hand_connections = animation["hand_connections"]

        word_markers.append({"word": word, "start_frame": len(all_frames)})

        all_frames.extend(animation["frames"])

        if len(words) > 1:
            last_frame = animation["frames"][-1]
            for _ in range(TRANSITION_FRAMES):
                all_frames.append(last_frame)

    if not all_frames:
        return jsonify(
            success=False, error="No se encontraron animaciones para esas palabras."
        )

    return jsonify(
        success=True,
        frames=all_frames,
        pose_connections=pose_connections,
        hand_connections=hand_connections,
        word_markers=word_markers,
        total_frames=len(all_frames),
    )
