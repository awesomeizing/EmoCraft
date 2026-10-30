"""
Smoke test for EmoCraft — runs a quick end-to-end check without full training.

Checks:
  1. Python environment and required packages
  2. FER2013 dataset structure and a small sample (10 images per class)
  3. YuNet face detector download and functionality
  4. Tiny model forward pass (untrained, just verifies shapes/dtypes)
  5. CREMA-D availability check (non-blocking)
  6. RAVDESS availability check

Usage:
    python smoke_test.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"  {status}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    return ok


def warn(label: str, detail: str = "") -> None:
    print(f"  {WARN}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def info(label: str, detail: str = "") -> None:
    print(f"  {INFO}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


all_passed = True


# ── 1. Python environment ─────────────────────────────────────────────────────
section("1. Python environment")

import sys
check("Python version", True, f"{sys.version}")

REQUIRED_PACKAGES = [
    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
]

for mod, pkg in REQUIRED_PACKAGES:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "unknown")
        check(f"{pkg} ({mod})", True, f"version {ver}")
    except ImportError as e:
        all_passed = False
        check(f"{pkg} ({mod})", False, f"NOT FOUND — install with: pip install {pkg}")


# ── 2. FER2013 dataset ────────────────────────────────────────────────────────
section("2. FER2013 dataset")

from emocraft import config
from emocraft.fer2013_data import verify_fer2013, scan_split

fer_dir = config.FER2013_DIR
problems = verify_fer2013(fer_dir)
if problems:
    all_passed = False
    check("FER2013 dataset structure", False, "\n".join(problems))
else:
    check("FER2013 dataset structure", True, str(fer_dir))
    # Sample count check
    for split in ("train", "test"):
        scan = scan_split(fer_dir / split, config.FER2013_CLASSES, verify=False)
        counts = {cls: len(files) for cls, files in scan.items()}
        total = sum(counts.values())
        check(f"FER2013 {split}/ file counts", total > 0,
              f"total={total}  per-class={counts}")

    # Quick image decode test (first 3 images per class in train/)
    from PIL import Image
    bad = []
    for cls in config.FER2013_CLASSES:
        cls_dir = fer_dir / "train" / cls
        for p in sorted(cls_dir.iterdir())[:3]:
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(p) as im:
                    arr = im.convert("L")
                    w, h = arr.size
                    if w != 48 or h != 48:
                        bad.append(f"{p.name}: {w}×{h} (expected 48×48)")
            except Exception as e:
                bad.append(f"{p.name}: {e}")
    if bad:
        warn("Some images have unexpected size or decode errors", "\n".join(bad[:5]))
    else:
        check("FER2013 sample image decode (48×48 grayscale)", True, "First 3 per class OK")


# ── 3. YuNet face detector ────────────────────────────────────────────────────
section("3. YuNet face detector")

import urllib.request
import numpy as np
import cv2

face_model_path = config.FACE_DETECTOR_MODEL
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

if not face_model_path.exists():
    print(f"  {INFO}  Downloading YuNet ONNX model ...")
    try:
        url = ("https://github.com/opencv/opencv_zoo/raw/main/"
               "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
        urllib.request.urlretrieve(url, face_model_path)
        check("YuNet model download", True, f"{face_model_path} ({face_model_path.stat().st_size:,} bytes)")
    except Exception as e:
        all_passed = False
        check("YuNet model download", False, str(e))

if face_model_path.exists():
    try:
        from emocraft.face_detector import detect_faces
        # Create a synthetic face-like test image (white square on grey background)
        test_img = np.full((200, 200, 3), 128, dtype=np.uint8)
        cv2.rectangle(test_img, (60, 40), (140, 160), (220, 185, 145), -1)  # skin-coloured box
        boxes = detect_faces(test_img, face_model_path)
        # Note: may not detect a synthetic face; just check it does not crash
        check("YuNet detector runs without crash", True,
              f"Detected {len(boxes)} faces in synthetic test image")
    except Exception as e:
        all_passed = False
        check("YuNet detector", False, traceback.format_exc())


# ── 4. Model architecture smoke test ─────────────────────────────────────────
section("4. Model architecture forward pass (untrained)")

try:
    import numpy as np
    from emocraft.model import build_emotion_cnn

    model = build_emotion_cnn(num_classes=7, image_size=48)
    x = np.random.rand(4, 48, 48, 1).astype(np.float32)
    probs = model.predict(x, verbose=0)
    ok = probs.shape == (4, 7) and abs(probs.sum(axis=1).mean() - 1.0) < 0.01
    check("Model forward pass", ok,
          f"Input shape (4, 48, 48, 1) → output shape {probs.shape}, row-sums ≈ 1.0: {ok}")
    total_params = model.count_params()
    check("Model parameter count", total_params > 0, f"{total_params:,} parameters")
except Exception as e:
    all_passed = False
    check("Model forward pass", False, traceback.format_exc())


# ── 5. Inference pipeline ─────────────────────────────────────────────────────
section("5. Inference pipeline (untrained model — only shape/type check)")

try:
    import numpy as np
    import cv2
    from emocraft.model import build_emotion_cnn
    from emocraft.inference import predict_image

    dummy_model = build_emotion_cnn(num_classes=7, image_size=48)
    dummy_meta = {"class_names": config.FER2013_CLASSES, "image_size": 48}

    # Create a minimal JPEG of a solid-grey face area
    img = np.full((100, 100, 3), 180, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    img_bytes = buf.tobytes()

    result = predict_image(img_bytes, dummy_model, dummy_meta, face_model_path)
    # May or may not find a face; just check it doesn't crash and returns expected keys
    required_keys = {"label", "confidence", "all_scores", "face_found", "error"}
    missing_keys = required_keys - set(result.keys())
    check("predict_image returns expected keys", not missing_keys,
          f"Missing keys: {missing_keys}" if missing_keys else "all required keys present")
    if result["error"] is None:
        check("predict_image label in class_names",
              result["label"] in config.FER2013_CLASSES,
              f"Got label: {result['label']}")
    else:
        info("predict_image returned an error (no face in synthetic image)", result["error"])
except Exception as e:
    all_passed = False
    check("Inference pipeline", False, traceback.format_exc())


# ── 6. CREMA-D check ──────────────────────────────────────────────────────────
section("6. CREMA-D dataset (video model training dataset)")

from emocraft.cremad_data import verify_cremad
cremad_ok, cremad_problems = verify_cremad(config.CREMAD_DIR)
if cremad_ok:
    check("CREMA-D available", True, str(config.CREMAD_DIR))
else:
    warn("CREMA-D not available (video model will fall back to RAVDESS demo)",
         "\n".join(cremad_problems[:3]))
    info("How to get CREMA-D:",
         "git clone https://github.com/CheyneyComputerScience/CREMA-D\n"
         "cd CREMA-D && git lfs pull\n"
         "Then set EMOCRAFT_CREMAD_DIR=<path to CREMA-D clone>")


# ── 7. RAVDESS check ──────────────────────────────────────────────────────────
section("7. RAVDESS dataset (pilot/smoke-test fallback)")

ravdess_dir = config.RAVDESS_DIR
archives = sorted(ravdess_dir.glob("Video_Speech_Actor_*.zip")) if ravdess_dir.is_dir() else []
if archives:
    check("RAVDESS ZIPs found", True,
          f"{len(archives)} archive(s): {[a.name for a in archives]}")
    warn("RAVDESS: only 2 actors (05, 06) downloaded",
         "Results from 2-actor training are NOT reliable generalization estimates.")
else:
    warn("RAVDESS not found", f"Expected ZIPs in {ravdess_dir}")


# ── 8. Existing pilot artefacts ───────────────────────────────────────────────
section("8. Existing pilot artefacts (RAVDESS demo run)")

run_summary = ravdess_dir / "run_output" / "run_summary.json"
if run_summary.exists():
    import json
    s = json.loads(run_summary.read_text())
    info("Previous RAVDESS pilot run summary",
         f"Clips: {s.get('compatible_clips')}  "
         f"Accuracy: {s.get('clip_accuracy', 0):.3f}  "
         f"Balanced accuracy: {s.get('clip_balanced_accuracy', 0):.3f}  "
         f"Macro-F1: {s.get('clip_macro_f1', 0):.3f}")
    warn("Prior pilot accuracy is very low (≈ 0.08)", s.get("warning", ""))
else:
    info("No previous RAVDESS pilot run found")


# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
if all_passed:
    print(f"  {PASS}  All required checks passed.\n")
    print("  Next steps:")
    print("    1.  python train_image.py            # train image model on FER2013")
    print("    2.  python train_video.py            # train video model (CREMA-D preferred)")
    print("    3.  streamlit run app.py             # launch the web app")
else:
    print(f"  {FAIL}  Some checks failed. Fix the issues above before training.\n")
    sys.exit(1)
