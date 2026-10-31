"""
EmoCraft project configuration.

All dataset paths, hyperparameters, and paths are defined here.
Override by setting EMOCRAFT_* environment variables or by editing
this file. Do NOT hard-code Windows-specific absolute paths outside
this module.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Project root (this file lives at <project_root>/emocraft/config.py) ─────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset paths ────────────────────────────────────────────────────────────
# FER2013: folder that contains train/ and test/ subdirs.
# Default: <project_root>/datasets/fer2013/
FER2013_DIR: Path = Path(
    os.environ.get("EMOCRAFT_FER2013_DIR",
                   str(PROJECT_ROOT / "datasets" / "fer2013"))
).expanduser().resolve()

# CREMA-D: AudioVisual/ subfolder within the CREMA-D repo clone.
# Default: <project_root>/datasets/cremad/
CREMAD_DIR: Path = Path(
    os.environ.get("EMOCRAFT_CREMAD_DIR",
                   str(PROJECT_ROOT / "datasets" / "cremad"))
).expanduser().resolve()

# RAVDESS: folder containing Video_Speech_Actor_*.zip files.
RAVDESS_DIR: Path = Path(
    os.environ.get("EMOCRAFT_RAVDESS_DIR",
                   str(PROJECT_ROOT / "RAVDESS"))
).expanduser().resolve()

# ── Model / output paths ─────────────────────────────────────────────────────
MODELS_DIR: Path = Path(
    os.environ.get("EMOCRAFT_MODELS_DIR",
                   str(PROJECT_ROOT / "models"))
).expanduser().resolve()

OUTPUTS_DIR: Path = Path(
    os.environ.get("EMOCRAFT_OUTPUTS_DIR",
                   str(PROJECT_ROOT / "outputs"))
).expanduser().resolve()

FACE_DETECTOR_MODEL: Path = Path(
    os.environ.get("EMOCRAFT_FACE_MODEL",
                   str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"))
).expanduser().resolve()

HSE_EXPRESSION_MODEL: Path = Path(
    os.environ.get(
        "EMOCRAFT_HSE_EXPRESSION_MODEL",
        str(MODELS_DIR / "enet_b0_8_best_afew.onnx"),
    )
).expanduser().resolve()

# ── Training hyperparameters ─────────────────────────────────────────────────
IMAGE_SIZE: int = int(os.environ.get("EMOCRAFT_IMAGE_SIZE", "48"))
BATCH_SIZE: int = int(os.environ.get("EMOCRAFT_BATCH_SIZE", "64"))
EPOCHS: int = int(os.environ.get("EMOCRAFT_EPOCHS", "50"))
LEARNING_RATE: float = float(os.environ.get("EMOCRAFT_LR", "1e-3"))
VAL_FRACTION: float = float(os.environ.get("EMOCRAFT_VAL_FRACTION", "0.10"))
EARLY_STOP_PATIENCE: int = int(os.environ.get("EMOCRAFT_PATIENCE", "8"))
LR_REDUCE_PATIENCE: int = int(os.environ.get("EMOCRAFT_LR_PATIENCE", "3"))
SEED: int = int(os.environ.get("EMOCRAFT_SEED", "42"))

# ── Video inference ──────────────────────────────────────────────────────────
VIDEO_FRAMES_PER_CLIP: int = int(os.environ.get("EMOCRAFT_VID_FRAMES", "16"))
VIDEO_SAMPLE_FPS: float = float(os.environ.get("EMOCRAFT_VID_FPS", "2.0"))

# ── FER2013 class labels (sorted, matches dataset folder names) ───────────────
FER2013_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── CREMA-D class labels (6 classes; no surprise/disgust split from FER) ─────
CREMAD_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# ── Label alignment note ─────────────────────────────────────────────────────
# FER2013 has 7 classes; CREMA-D has 6 (no 'surprise').
# The image model is trained on FER2013's 7 classes.
# The video model (if CREMA-D is used) is trained on CREMA-D's 6 classes.
# The Streamlit app loads whichever model was trained and reads its own class
# list from the saved label map — no label remapping is done silently.

