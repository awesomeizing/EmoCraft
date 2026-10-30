"""
CREMA-D video dataset utilities.

CREMA-D (Crowd Sourced Emotional Multimodal Actors Dataset)
  Source:   https://github.com/CheyneyComputerScience/CREMA-D
  License:  Open Database License (ODbL) v1.0
  Size:     ~8.5 GB (full dataset via Git LFS)
  Classes:  6  (ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad)

Filename format: <actor_id>_<sentence_id>_<emotion>_<intensity>.flv
Example: 1001_DFA_ANG_XX.flv

This module:
  1. Verifies whether CREMA-D is genuinely present (real media vs LFS stubs).
  2. Scans the AudioVisual/ subdirectory for .flv / .mp4 files.
  3. Builds actor-disjoint train/val/test splits (test actors never seen in training).
  4. Extracts face crops from video frames using YuNet.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from emocraft import config

# CREMA-D emotion code → canonical label
CREMAD_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

_CREMAD_CLASS_NAMES = config.CREMAD_CLASSES  # ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Minimum real file size in bytes (LFS stub files are tiny, typically < 300 bytes)
_LFS_STUB_THRESHOLD = 1_000


def is_lfs_stub(path: Path) -> bool:
    """Return True if the file looks like a Git LFS pointer stub (not real media)."""
    if not path.is_file():
        return True
    if path.stat().st_size < _LFS_STUB_THRESHOLD:
        return True
    # Check for the LFS pointer header
    try:
        with path.open("rb") as f:
            header = f.read(64)
        return b"version https://git-lfs" in header
    except Exception:
        return False


def verify_cremad(cremad_dir: Path) -> tuple[bool, list[str]]:
    """
    Check whether CREMA-D is available and usable.

    Returns (is_usable, problems).
    is_usable = True only when real media files are found.
    """
    problems = []
    if not cremad_dir.is_dir():
        problems.append(
            f"CREMA-D directory not found: {cremad_dir}\n"
            "  → Clone https://github.com/CheyneyComputerScience/CREMA-D (requires Git LFS)\n"
            "    and place/symlink the repo so that AudioVisual/ is inside it."
        )
        return False, problems

    av_dir = _find_av_dir(cremad_dir)
    if av_dir is None:
        problems.append(
            f"Could not find AudioVisual/ directory inside {cremad_dir}.\n"
            "  Expected: <cremad_dir>/AudioVisual/*.flv  or  <cremad_dir>/VideoFlash/*.flv"
        )
        return False, problems

    # Check for real media files (not LFS stubs)
    media_files = list(av_dir.glob("*.flv")) + list(av_dir.glob("*.mp4"))
    if not media_files:
        problems.append(f"No .flv or .mp4 files found in {av_dir}")
        return False, problems

    real_count = sum(1 for f in media_files[:20] if not is_lfs_stub(f))
    if real_count == 0:
        problems.append(
            f"All sampled files in {av_dir} appear to be Git LFS pointer stubs.\n"
            "  → Run `git lfs pull` inside the CREMA-D repo to download real media files.\n"
            "  → Note: the full dataset requires ~8.5 GB of disk space."
        )
        return False, problems

    return True, problems


def _find_av_dir(cremad_dir: Path) -> Optional[Path]:
    """Locate the directory containing the CREMA-D video files."""
    for candidate in ("AudioVisual", "VideoFlash", "Video"):
        d = cremad_dir / candidate
        if d.is_dir():
            return d
    # Search one level deeper
    for child in cremad_dir.iterdir():
        if not child.is_dir():
            continue
        for candidate in ("AudioVisual", "VideoFlash", "Video"):
            d = child / candidate
            if d.is_dir():
                return d
    return None


def parse_cremad_filename(fname: str) -> Optional[dict]:
    """
    Parse a CREMA-D filename into metadata.
    Returns None if the filename does not match the expected pattern.

    Format: <actor_id>_<sentence_id>_<emotion>_<intensity>.ext
    Example: 1001_DFA_ANG_XX.flv  → actor=1001, emotion=angry
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    actor_id = parts[0]
    emotion_code = parts[2].upper()
    label = CREMAD_EMOTION_MAP.get(emotion_code)
    if label is None:
        return None
    return {"actor": actor_id, "label": label, "filename": fname}


def scan_cremad(cremad_dir: Path) -> list[dict]:
    """
    Scan CREMA-D and return a list of record dicts for every valid, real-media clip.
    Each record has: actor, label, filename, path.
    """
    av_dir = _find_av_dir(cremad_dir)
    if av_dir is None:
        return []

    records = []
    for path in sorted(av_dir.glob("*.flv")) + sorted(av_dir.glob("*.mp4")):
        if is_lfs_stub(path):
            continue
        parsed = parse_cremad_filename(path.name)
        if parsed is None:
            continue
        records.append({**parsed, "path": path})

    return records


def actor_disjoint_split(
    records: list[dict],
    test_actor_fraction: float = 0.15,
    val_actor_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split CREMA-D clips by actor identity so no actor appears in more than one set.

    Returns (train_records, val_records, test_records).
    """
    rng = random.Random(seed)
    actors = sorted({r["actor"] for r in records})
    n = len(actors)
    if n < 3:
        raise RuntimeError(
            f"Need at least 3 actors for a split, found {n}. "
            "CREMA-D has 91 actors; check that real media files were downloaded."
        )
    n_test = max(1, int(n * test_actor_fraction))
    n_val = max(1, int(n * val_actor_fraction))

    shuffled = list(actors)
    rng.shuffle(shuffled)
    test_actors = set(shuffled[:n_test])
    val_actors = set(shuffled[n_test:n_test + n_val])

    train = [r for r in records if r["actor"] not in test_actors and r["actor"] not in val_actors]
    val = [r for r in records if r["actor"] in val_actors]
    test = [r for r in records if r["actor"] in test_actors]
    return train, val, test


def extract_face_crops(
    records: list[dict],
    face_model_path: Path,
    image_size: int = 48,
    frames_per_clip: int = 8,
) -> tuple[list[dict], list[dict]]:
    """
    For each clip record, sample frames and extract face crops.

    Returns (usable_records, failed_records).
    Each usable record gains a 'crops' key: list of float32 arrays (H, W).
    """
    from emocraft.face_detector import detect_faces

    usable, failed = [], []
    for rec in records:
        cap = cv2.VideoCapture(str(rec["path"]))
        if not cap.isOpened():
            failed.append({**rec, "reason": "could not open video"})
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
                    continue
                x, y, fw, fh = boxes[0]
                mx, my = int(0.15 * fw), int(0.15 * fh)
                x1, y1 = max(0, x - mx), max(0, y - my)
                x2, y2 = min(bgr.shape[1], x + fw + mx), min(bgr.shape[0], y + fh + my)
                face = bgr[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                crop = cv2.resize(gray, (image_size, image_size),
                                  interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                crops.append(crop)
            if not crops:
                failed.append({**rec, "reason": "no face found in any sampled frame"})
                continue
            usable.append({**rec, "crops": crops})
        finally:
            cap.release()
    return usable, failed

