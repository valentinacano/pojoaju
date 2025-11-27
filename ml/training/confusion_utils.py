import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import load_model

from ml.utils.training_utils import get_sequences_and_labels
from app.database.database_utils import fetch_word_ids_with_keypoints, search_word_id
from app.config import MODEL_FRAMES, MODEL_PATH


def generate_confusion_matrix(save_path="static/confusion/confusion_matrix.png"):
    """
    Genera matriz de confusión con labels de palabras reales y
    la guarda en static/confusion/
    
    Returns:
        tuple: (cm, y_val, y_pred, metrics_dict) donde metrics_dict contiene
               accuracy, report y otros datos útiles para visualización
    """
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("📌 Obteniendo words...")
    word_ids = fetch_word_ids_with_keypoints()
    
    if len(word_ids) < 2:
        raise ValueError("Se necesitan al menos 2 palabras con keypoints para generar la matriz")

    # Obtener nombres REALES por cada ID
    idx_to_word = {}
    for i, wid in enumerate(word_ids):
        rec = search_word_id(wid)
        if rec:
            _, palabra, _ = rec
            idx_to_word[i] = palabra
        else:
            idx_to_word[i] = f"word_{i}"

    print("📌 Obteniendo secuencias y labels...")
    sequences, labels = get_sequences_and_labels(word_ids)

    sequences = pad_sequences(
        sequences,
        maxlen=int(MODEL_FRAMES),
        padding="pre",
        truncating="post",
        dtype="float16",
    )

    X = np.array(sequences)
    y = np.array(labels)

    # Usar un test_size más grande para tener mejor representación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.8, random_state=42, stratify=y
    )

    print(f"📌 Validación: {len(X_val)} muestras")
    print("📌 Cargando modelo...")
    model = load_model(MODEL_PATH)

    print("📌 Prediciendo validation set...")
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("📌 Calculando matriz...")
    cm = confusion_matrix(y_val, y_pred)

    # Solo clases que aparecen en validación
    unique_classes = sorted(list(set(y_val)))
    labels_text = [idx_to_word[i] for i in unique_classes]

    # Calcular accuracy
    accuracy = np.sum(y_val == y_pred) / len(y_val)
    
    # Reporte de clasificación
    report = classification_report(
        y_val, 
        y_pred, 
        target_names=labels_text,
        output_dict=True,
        zero_division=0
    )

    # Crear figura más grande si hay muchas clases
    n_classes = len(unique_classes)
    figsize = (max(10, n_classes * 0.6), max(8, n_classes * 0.5))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_text)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
    
    plt.title(f"Matriz de Confusión — Modelo LSPy\nAccuracy: {accuracy:.2%}", 
              fontsize=14, pad=20)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Valor Real", fontsize=12)
    
    # Ajustar layout para evitar cortes
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Matriz guardada en: {save_path}")
    print(f"✅ Accuracy: {accuracy:.2%}")

    # Retornar métricas útiles
    metrics = {
        'accuracy': accuracy,
        'report': report,
        'n_classes': n_classes,
        'n_samples': len(y_val),
        'labels': labels_text
    }

    return cm, y_val, y_pred, metrics


def get_top_confusions(cm, labels_text, top_k=5):
    """
    Encuentra los top K pares de palabras más confundidos
    
    Args:
        cm: matriz de confusión (numpy array)
        labels_text: lista con nombres de las clases
        top_k: número de confusiones a retornar
        
    Returns:
        list: lista de tuplas (palabra_real, palabra_predicha, cantidad)
    """
    confusions = []
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:  # Solo off-diagonal (errores)
                confusions.append((
                    labels_text[i],  # real
                    labels_text[j],  # predicha
                    int(cm[i, j])    # cantidad
                ))
    
    # Ordenar por cantidad descendente
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    return confusions[:top_k]