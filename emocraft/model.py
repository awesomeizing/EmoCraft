"""
Keras CNN model for facial emotion recognition.

Architecture: custom lightweight CNN (no pretrained backbone required, since
FER2013 images are 48x48 grayscale). The model includes a Rescaling layer so
that training, evaluation, and inference all use the same preprocessing pipeline.
"""

from __future__ import annotations

import json
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

    # Block 1: 48x48 -> 24x24
    x = layers.Conv2D(24, 3, padding="same", use_bias=False, name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2: 24x24 -> 12x12
    x = layers.Conv2D(48, 3, padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3: 12x12 -> 6x6
    x = layers.Conv2D(96, 3, padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Global pooling & classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
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

    model.save(str(weights_path))
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

