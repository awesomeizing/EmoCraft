"""
Face detection utility using OpenCV's YuNet detector (FaceDetectorYN).

YuNet is bundled with OpenCV >= 4.8 via FaceDetectorYN but requires an ONNX
model file. The model is downloaded automatically on first use if not present.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

_detector_cache: dict[str, cv2.FaceDetectorYN] = {}


def _ensure_model(model_path: Path) -> Path:
    """Download YuNet ONNX if it is missing."""
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[face_detector] Downloading YuNet model to {model_path} ...")
        urllib.request.urlretrieve(_YUNET_URL, model_path)
        print(f"[face_detector] Downloaded ({model_path.stat().st_size:,} bytes)")
    return model_path


def get_detector(model_path: Path, input_hw: tuple[int, int] = (300, 300)) -> cv2.FaceDetectorYN:
    """Return a cached FaceDetectorYN for the given model path and input size."""
    key = f"{model_path}|{input_hw}"
    if key not in _detector_cache:
        _ensure_model(model_path)
        h, w = input_hw
        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (w, h),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        _detector_cache[key] = detector
    return _detector_cache[key]


def detect_faces(bgr: np.ndarray, model_path: Path) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in a BGR image.

    Returns a list of (x, y, w, h) bounding boxes in descending area order.
    Returns an empty list if no face is found or the image is invalid.
    """
    if bgr is None or bgr.size == 0:
        return []

    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return []

    detector = get_detector(model_path, (h, w))
    detector.setInputSize((w, h))

    _, faces = detector.detect(bgr)

    if faces is None or len(faces) == 0:
        return []

    boxes = []
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        # clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)
        if fw > 0 and fh > 0:
            boxes.append((x, y, fw, fh))

    # sort by area descending (largest face first)
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def crop_face(bgr: np.ndarray, model_path: Path,
              target_size: int = 48) -> Optional[np.ndarray]:
    """
    Detect the largest face, crop it, convert to grayscale, resize to
    target_size × target_size, and return a float32 array in [0, 1].

    Returns None if no face is detected.
    """
    boxes = detect_faces(bgr, model_path)
    if not boxes:
        return None

    x, y, fw, fh = boxes[0]
    # add a small margin (15 %) so the crop includes forehead / chin
    margin_x = int(0.15 * fw)
    margin_y = int(0.15 * fh)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(bgr.shape[1], x + fw + margin_x)
    y2 = min(bgr.shape[0], y + fh + margin_y)

    face_bgr = bgr[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return None

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

