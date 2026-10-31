"""
Inference utilities for EmoCraft image and video emotion recognition.
"""

from __future__ import annotations

import io
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Image inference ───────────────────────────────────────────────────────────

def predict_image(
    image_bytes: bytes,
    model,
    meta: dict,
    face_model_path: Path,
) -> dict:
    """
    Run emotion prediction on a single image supplied as raw bytes.

    Returns a dict with keys:
      - 'label': predicted emotion string
      - 'confidence': float in [0, 1]
      - 'all_scores': {class: score} dict
      - 'face_found': bool
      - 'annotated_bgr': annotated BGR image as np.ndarray (for display)
      - 'error': None or error message string
    """
    from emocraft.face_detector import crop_face, detect_faces

    class_names: list[str] = meta["class_names"]
    image_size: int = meta["image_size"]

    # Decode the image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return _error_result("Could not decode image. Please upload a valid JPEG or PNG.", class_names)

    # Detect faces
    boxes = detect_faces(bgr, face_model_path)

    if not boxes:
        # No face found — try treating the full image as a pre-cropped face
        # (handles 48×48 FER-style crops and passport-style close-ups)
        h, w = bgr.shape[:2]
        if max(h, w) <= 128:
            # Assume pre-cropped
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            crop_f32 = cv2.resize(gray, (image_size, image_size),
                                  interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            annotated = bgr.copy()
            cv2.putText(annotated, "[pre-cropped]", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            face_found = False
        else:
            return {**_error_result("No face detected in the image.", class_names),
                    "annotated_bgr": bgr.copy()}
    else:
        x, y, fw, fh = boxes[0]
        # margin
        mx = int(0.15 * fw)
        my = int(0.15 * fh)
        x1 = max(0, x - mx);  y1 = max(0, y - my)
        x2 = min(bgr.shape[1], x + fw + mx)
        y2 = min(bgr.shape[0], y + fh + my)
        face_bgr = bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        crop_f32 = cv2.resize(gray, (image_size, image_size),
                               interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        annotated = bgr.copy()
        cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 220, 0), 2)
        face_found = True

    # Model inference
    tensor = crop_f32[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)
    probs = _predict_probabilities(model, crop_f32)    # (num_classes,)

    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = float(probs[idx])
    all_scores = {c: float(probs[i]) for i, c in enumerate(class_names)}

    # Annotate
    text = f"{label} ({confidence:.0%})"
    if face_found:
        x_text = boxes[0][0]
        y_text = max(boxes[0][1] - 8, 15)
    else:
        x_text, y_text = 5, 20
    cv2.putText(annotated, text, (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)

    return {
        "label": label,
        "confidence": confidence,
        "all_scores": all_scores,
        "face_found": face_found,
        "annotated_bgr": annotated,
        "error": None,
    }


# ── Video inference ───────────────────────────────────────────────────────────

def predict_video(
    video_path: Path,
    model,
    meta: dict,
    face_model_path: Path,
    target_fps: float = 2.0,
    max_frames: int = 60,
) -> dict:
    """
    Sample frames from a video, run per-frame inference, and aggregate.

    Returns a dict with keys:
      - 'label': majority-vote emotion
      - 'confidence': mean confidence of the majority class
      - 'all_scores': mean per-class scores across frames
      - 'per_frame': list of {frame_idx, label, confidence} for each processed frame
      - 'frames_processed': int
      - 'frames_with_face': int
      - 'annotated_frames_bgr': list of annotated BGR frames
      - 'error': None or error message string
    """
    from emocraft.face_detector import detect_faces

    class_names: list[str] = meta["class_names"]
    image_size: int = meta["image_size"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _video_error(f"Could not open video: {video_path}", class_names)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # fallback

        if total_frames < 1:
            return _video_error("Video appears to have no frames.", class_names)

        # Choose which frames to sample
        step = max(1, int(fps / target_fps))
        candidate_indices = np.arange(0, total_frames, step, dtype=np.int64)
        if len(candidate_indices) > max_frames:
            sample_positions = np.linspace(0, len(candidate_indices) - 1,
                                           max_frames).round().astype(int)
            frame_indices = candidate_indices[sample_positions].tolist()
        else:
            frame_indices = candidate_indices.tolist()

        per_frame = []
        annotated_frames = []
        accumulated_probs = np.zeros(len(class_names), dtype=np.float64)
        frames_with_face = 0

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue

            boxes = detect_faces(bgr, face_model_path)
            annotated = bgr.copy()

            if boxes:
                x, y, fw, fh = boxes[0]
                mx = int(0.15 * fw);  my = int(0.15 * fh)
                x1 = max(0, x - mx);  y1 = max(0, y - my)
                x2 = min(bgr.shape[1], x + fw + mx)
                y2 = min(bgr.shape[0], y + fh + my)
                face_bgr = bgr[y1:y2, x1:x2]
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                crop_f32 = cv2.resize(gray, (image_size, image_size),
                                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                probs = _predict_probabilities(model, crop_f32)
                idx = int(np.argmax(probs))
                label = class_names[idx]
                conf = float(probs[idx])

                accumulated_probs += probs
                frames_with_face += 1

                per_frame.append({"frame_idx": fi, "label": label, "confidence": conf})

                # Annotate frame
                cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 220, 0), 2)
                cv2.putText(annotated, f"{label} {conf:.0%}", (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(annotated, "no face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)

            annotated_frames.append(annotated)

        if frames_with_face == 0:
            return {**_video_error("No faces detected in any sampled frame.", class_names),
                    "per_frame": per_frame,
                    "frames_processed": len(frame_indices),
                    "frames_with_face": 0,
                    "annotated_frames_bgr": annotated_frames}

        # Aggregate
        mean_probs = accumulated_probs / frames_with_face
        best_idx = int(np.argmax(mean_probs))
        label = class_names[best_idx]
        confidence = float(mean_probs[best_idx])
        all_scores = {c: float(mean_probs[i]) for i, c in enumerate(class_names)}

        return {
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores,
            "per_frame": per_frame,
            "frames_processed": len(frame_indices),
            "frames_with_face": frames_with_face,
            "annotated_frames_bgr": annotated_frames,
            "error": None,
        }
    finally:
        cap.release()


def write_annotated_video(
    annotated_frames: list[np.ndarray],
    output_path: Path,
    fps: float = 2.0,
) -> Optional[Path]:
    """
    Write annotated frames to an MP4 video. Returns the path or None on failure.
    """
    if not annotated_frames:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = annotated_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not out.isOpened():
        return None
    for frame in annotated_frames:
        out.write(frame)
    out.release()
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_result(msg: str, class_names: list[str]) -> dict:
    return {
        "label": None,
        "confidence": 0.0,
        "all_scores": {c: 0.0 for c in class_names},
        "face_found": False,
        "annotated_bgr": None,
        "error": msg,
    }


def _predict_probabilities(model, crop_f32: np.ndarray) -> np.ndarray:
    """Average predictions for a face crop and its horizontal mirror."""
    crop = np.asarray(crop_f32, dtype=np.float32)
    variants = np.stack((crop, np.fliplr(crop)), axis=0)[..., np.newaxis]
    probabilities = model.predict(variants, verbose=0)
    return np.asarray(probabilities, dtype=np.float32).mean(axis=0)


def _video_error(msg: str, class_names: list[str]) -> dict:
    return {
        "label": None,
        "confidence": 0.0,
        "all_scores": {c: 0.0 for c in class_names},
        "per_frame": [],
        "frames_processed": 0,
        "frames_with_face": 0,
        "annotated_frames_bgr": [],
        "error": msg,
    }

