"""
Keras CNN model for facial emotion recognition.

Architecture: residual CNN for 48x48 grayscale facial crops. Training-time
augmentation is embedded in the model so saved models use the same input
contract at training and inference; augmentation layers are inactive at inference.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


def build_emotion_cnn(
    num_classes: int,
    image_size: int = 48,
) -> "keras.Model":
    """
    Build a CNN for emotion recognition.
    Input: (batch, image_size, image_size, 1), float32 in [0, 1].
    Output: (batch, num_classes) logits (softmax applied separately).
    """
    try:
        import keras
        from keras import layers
    except ImportError:
        raise ImportError("Keras/TensorFlow is required. pip install tensorflow")

    inp = keras.Input(shape=(image_size, image_size, 1), name="image_input")

    x = layers.RandomFlip("horizontal", name="augment_flip")(inp)
    x = layers.RandomRotation(0.04, name="augment_rotation")(x)
    x = layers.RandomTranslation(0.06, 0.06, name="augment_translation")(x)
    x = layers.RandomZoom(0.08, name="augment_zoom")(x)
    x = layers.RandomContrast(0.12, name="augment_contrast")(x)

    # Residual blocks retain facial layout while learning richer expression features.
    for block, (filters, dropout) in enumerate(
        ((16, 0.10), (32, 0.15), (64, 0.20), (96, 0.25)), start=1
    ):
        shortcut = x
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          name=f"block{block}_conv1")(x)
        x = layers.BatchNormalization(name=f"block{block}_bn1")(x)
        x = layers.Activation("relu", name=f"block{block}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          name=f"block{block}_conv2")(x)
        x = layers.BatchNormalization(name=f"block{block}_bn2")(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                     name=f"block{block}_projection")(shortcut)
            shortcut = layers.BatchNormalization(name=f"block{block}_projection_bn")(shortcut)
        x = layers.Add(name=f"block{block}_add")([x, shortcut])
        x = layers.Activation("relu", name=f"block{block}_relu2")(x)
        x = layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
        x = layers.SpatialDropout2D(dropout, name=f"block{block}_dropout")(x)

    # Combining average and max pooling keeps both broad and salient expression cues.
    avg = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    max_pool = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Concatenate(name="pool_concat")([avg, max_pool])
    x = layers.Dense(128, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Activation("relu", name="fc_relu")(x)
    x = layers.Dropout(0.40, name="fc_dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="EmoCraftCNN")
    return model


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_model(
    model: "keras.Model",
    class_names: list[str],
    image_size: int,
    save_dir: Path,
    model_stem: str = "emocraft_image",
) -> tuple[Path, Path]:
    """
    Save the Keras model weights (.keras) and a metadata JSON alongside it.
    Returns (weights_path, meta_path).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    weights_path = save_dir / f"{model_stem}.keras"
    meta_path = save_dir / f"{model_stem}_meta.json"

    # Keras writes a temporary archive before replacing the destination. Keep
    # that temporary file inside the writable model directory on restricted
    # Windows environments, where the system temp directory may be blocked.
    temp_dir = save_dir / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    old_tempdir = tempfile.tempdir
    tempfile.tempdir = str(temp_dir)
    try:
        model.save(str(weights_path))
    finally:
        tempfile.tempdir = old_tempdir
    meta = {
        "class_names": class_names,
        "image_size": image_size,
        "num_classes": len(class_names),
        "model_stem": model_stem,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[model] Saved weights -> {weights_path}")
    print(f"[model] Saved metadata -> {meta_path}")
    return weights_path, meta_path


def load_model(
    save_dir: Path,
    model_stem: str = "emocraft_image",
) -> tuple["keras.Model", dict]:
    """
    Load a saved EmoCraft model and its metadata.
    Returns (model, meta_dict).
    Raises FileNotFoundError if the files are missing.
    """
    try:
        import keras
    except ImportError:
        raise ImportError("Keras/TensorFlow is required. pip install tensorflow")

    save_dir = Path(save_dir)
    weights_path = save_dir / f"{model_stem}.keras"
    meta_path = save_dir / f"{model_stem}_meta.json"

    for p in (weights_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {p}\n"
                "  → Run training first with: python train_image.py"
            )

    model = keras.models.load_model(str(weights_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta

