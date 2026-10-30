"""
RAVDESS video dataset utilities (used for pilot/smoke-test when CREMA-D is absent).

⚠  WARNING: Only actors 05 and 06 have been downloaded.
   Two actors cannot provide reliable generalization estimates.
   This module is used only for a clearly labeled smoke-test / demo.
"""

from __future__ import annotations

import random
import zipfile
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from emocraft import config

# RAVDESS emotion code → FER2013-aligned label (calm is excluded — FER has no calm class)
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    # "02": "calm",  # no FER equivalent; excluded
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def _parse_name(filename: str) -> Optional[dict]:
    """Parse a RAVDESS filename like '02-01-05-01-01-02-06.mp4'."""
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "02" or channel != "01":
        return None  # video-only speech only
    label = RAVDESS_EMOTION_MAP.get(emotion)
    if label is None:
        return None
    return {"actor": actor, "label": label, "emotion_code": emotion, "filename": filename}


def _extract_archives(ravdess_dir: Path, extract_dir: Path) -> list[dict]:
    """Extract ZIP archives and return clip records."""
    archives = sorted(ravdess_dir.glob("Video_Speech_Actor_*.zip"))
    if not archives:
        return []
    extract_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() != ".mp4":
                    continue
                parsed = _parse_name(member.filename)
                if parsed is None:
                    continue
                actor_dir = extract_dir / f"Actor_{parsed['actor']}"
                actor_dir.mkdir(parents=True, exist_ok=True)
                dest = actor_dir / Path(member.filename).name
                if not dest.is_file() or dest.stat().st_size != member.file_size:
                    with zf.open(member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                records.append({**parsed, "path": dest})
    return records


def load_ravdess_records(ravdess_dir: Path) -> list[dict]:
    """
    Load all RAVDESS video-only speech records from the archive directory.
    Extracts ZIPs if needed.
    """
    extract_dir = ravdess_dir / "run_output" / "video_only_speech"
    # Try already-extracted files first
    records = []
    if extract_dir.is_dir():
        for mp4 in sorted(extract_dir.rglob("*.mp4")):
            parsed = _parse_name(mp4.name)
            if parsed is None:
                continue
            records.append({**parsed, "path": mp4})

    if not records:
        records = _extract_archives(ravdess_dir, extract_dir)

    records.sort(key=lambda r: (r["actor"], r["label"], r["filename"]))
    return records


def extract_face_crops(
    records: list[dict],
    face_model_path: Path,
    image_size: int = 48,
    frames_per_clip: int = 8,
) -> tuple[list[dict], list[dict]]:
    """Extract face crops using YuNet detector."""
    from emocraft.face_detector import detect_faces

    usable, failed = [], []
    for rec in records:
        cap = cv2.VideoCapture(str(rec["path"]))
        if not cap.isOpened():
            failed.append({**rec, "reason": "could not open"})
            continue
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames < 1:
                failed.append({**rec, "reason": "no frames"})
                continue
            indices = np.unique(
                np.linspace(0, n_frames - 1, min(frames_per_clip, n_frames)).round().astype(int)
            )
            crops = []
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                boxes = detect_faces(bgr, face_model_path)
                if not boxes:
                    # RAVDESS fallback: centered crop (fixed studio setup)
                    h, w = bgr.shape[:2]
                    x1, x2 = int(0.30 * w), int(0.70 * w)
                    y1, y2 = int(0.03 * h), int(0.78 * h)
                    face = bgr[y1:y2, x1:x2]
                else:
                    x, y, fw, fh = boxes[0]
                    mx, my = int(0.15 * fw), int(0.15 * fh)
                    face = bgr[max(0, y-my):min(bgr.shape[0], y+fh+my),
                               max(0, x-mx):min(bgr.shape[1], x+fw+mx)]
                if face.size == 0:
                    continue
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                crop = cv2.resize(gray, (image_size, image_size),
                                  interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                crops.append(crop)
            if not crops:
                failed.append({**rec, "reason": "no usable frame crops"})
                continue
            usable.append({**rec, "crops": crops})
        finally:
            cap.release()
    return usable, failed


def split_by_actor(
    records: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], str, str]:
    """
    Split by actor identity. With 2 actors: actor_05=train+val, actor_06=test.
    Returns (train_recs, val_recs, test_recs, train_actor_id, test_actor_id).
    """
    from sklearn.model_selection import train_test_split

    actors = sorted({r["actor"] for r in records})
    if len(actors) < 2:
        raise RuntimeError(f"Need at least 2 actors, found {len(actors)}")

    train_actor, test_actor = actors[0], actors[-1]
    trainval = [r for r in records if r["actor"] == train_actor]
    test = [r for r in records if r["actor"] == test_actor]

    labels = [r["label"] for r in trainval]
    try:
        train, val = train_test_split(trainval, test_size=0.20, random_state=seed, stratify=labels)
    except ValueError:
        train, val = train_test_split(trainval, test_size=0.20, random_state=seed)

    return train, val, test, train_actor, test_actor

