"""
FER2013 image dataset utilities.

The msambare/fer2013 Kaggle dataset has this layout after extraction:
    fer2013/
        train/
            angry/   disgust/  fear/  happy/  neutral/  sad/  surprise/
        test/
            angry/   disgust/  fear/  happy/  neutral/  sad/  surprise/

File naming convention:
    train/  → Training_NNNNNNN.jpg
    test/   → PrivateTest_NNNNNNN.jpg   (both PublicTest and PrivateTest
                                          land in the same test/ folder)

We keep the official train / test split intact. We hold out VAL_FRACTION of
the *training* set for validation (stratified by class). The *test* split is
only used once, at evaluation time.

Labels are the folder names, sorted alphabetically:
    ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from emocraft import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")


def _try_open(p: Path) -> bool:
    """Return True if PIL can open and verify the file."""
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def _load_as_array(p: Path, target_size: int) -> Optional[np.ndarray]:
    """
    Load an image as a grayscale float32 array of shape (target_size, target_size).
    Returns None on decode failure.
    """
    try:
        with Image.open(p) as im:
            im = im.convert("L")  # grayscale
            im = im.resize((target_size, target_size), Image.BILINEAR)
            return np.array(im, dtype=np.float32) / 255.0
    except Exception:
        return None


# ── Dataset scanning ──────────────────────────────────────────────────────────

def scan_split(
    split_dir: Path,
    class_names: list[str],
    verify: bool = False,
) -> dict[str, list[Path]]:
    """
    Return {class_name: [image_path]} for a given split directory.
    Unknown class folders are silently ignored.
    Unreadable files are excluded when verify=True.
    """
    result: dict[str, list[Path]] = {c: [] for c in class_names}
    skipped_corrupt = 0
    for cls in class_names:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if not _is_image(p):
                continue
            if verify and not _try_open(p):
                skipped_corrupt += 1
                continue
            result[cls].append(p)
    if skipped_corrupt:
        print(f"[fer2013] WARNING: {skipped_corrupt} unreadable image(s) excluded from '{split_dir.name}'")
    return result


def verify_fer2013(fer_dir: Path) -> list[str]:
    """
    Check that the FER2013 directory has the expected structure.
    Returns a list of problem strings; empty list means all is OK.
    """
    problems = []
    if not fer_dir.is_dir():
        problems.append(
            f"FER2013 directory not found: {fer_dir}\n"
            "  → Download from https://www.kaggle.com/datasets/msambare/fer2013\n"
            "    and extract so that train/ and test/ are inside it."
        )
        return problems

    for split in ("train", "test"):
        split_dir = fer_dir / split
        if not split_dir.is_dir():
            problems.append(f"Missing split directory: {split_dir}")
            continue
        found_classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        if not found_classes:
            problems.append(f"No class subdirectories found in {split_dir}")
            continue
        for cls in config.FER2013_CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                problems.append(f"Missing class folder: {cls_dir}")
            else:
                n_imgs = sum(1 for p in cls_dir.iterdir() if _is_image(p))
                if n_imgs == 0:
                    problems.append(f"Empty class folder (no images): {cls_dir}")
    return problems


# ── In-memory dataset loader (ultra-fast, bypasses Windows/OneDrive file-handle overhead) ──

def load_fer2013_tensors(
    fer_dir: Path,
    image_size: int = 48,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load FER2013 into memory tensors in seconds.
    Reads from fer2013.zip if available, or falls back to extracted directories.
    Splits train into train and val (stratified by class).
    test is held out and never used for training.

    Returns:
        train_x, train_y, val_x, val_y, test_x, test_y, class_names
    """
    import zipfile
    import cv2
    from sklearn.model_selection import train_test_split

    class_names = config.FER2013_CLASSES
    class_to_id = {c: i for i, c in enumerate(class_names)}
    zip_path = fer_dir / "fer2013.zip"

    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    if zip_path.is_file():
        print(f"[fer2013] Loading dataset directly from {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                if not member.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                parts = member.split("/")
                if len(parts) < 3:
                    continue
                split, cls = parts[0], parts[1]
                if cls not in class_to_id:
                    continue
                data = z.read(member)
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if img.shape[:2] != (image_size, image_size):
                    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

                if split == "train":
                    train_imgs.append(img)
                    train_labels.append(class_to_id[cls])
                elif split == "test":
                    test_imgs.append(img)
                    test_labels.append(class_to_id[cls])
    else:
        print(f"[fer2013] Loading dataset from extracted directories in {fer_dir} ...")
        for split in ("train", "test"):
            split_dir = fer_dir / split
            for cls in class_names:
                cls_dir = split_dir / cls
                if not cls_dir.is_dir():
                    continue
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    if img.shape[:2] != (image_size, image_size):
                        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

                    if split == "train":
                        train_imgs.append(img)
                        train_labels.append(class_to_id[cls])
                    else:
                        test_imgs.append(img)
                        test_labels.append(class_to_id[cls])

    if not train_imgs or not test_imgs:
        raise ValueError(f"Could not load images from {fer_dir}. Found train={len(train_imgs)}, test={len(test_imgs)}")

    raw_train_x = (np.array(train_imgs, dtype=np.float32) / 255.0)[..., np.newaxis]
    raw_train_y = np.array(train_labels, dtype=np.int32)
    test_x = (np.array(test_imgs, dtype=np.float32) / 255.0)[..., np.newaxis]
    test_y = np.array(test_labels, dtype=np.int32)

    train_x, val_x, train_y, val_y = train_test_split(
        raw_train_x,
        raw_train_y,
        test_size=val_fraction,
        random_state=seed,
        stratify=raw_train_y,
    )

    print(f"[fer2013] Loaded shapes: train={train_x.shape}, val={val_x.shape}, test={test_x.shape}")
    return train_x, train_y, val_x, val_y, test_x, test_y, class_names


# ── Keras data generators ─────────────────────────────────────────────────────

def make_keras_generators(
    fer_dir: Path,
    image_size: int = 48,
    batch_size: int = 64,
    val_fraction: float = 0.10,
    seed: int = 42,
    verify: bool = False,
):
    """
    Build Keras ImageDataGenerators for FER2013 train, val, and test.

    Returns (train_gen, val_gen, test_gen, class_names)

    - train/val are split from the official train folder (stratified).
    - test is the official test folder; it is never used for fitting.
    - val_fraction: fraction of train images withheld for validation.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        raise ImportError("TensorFlow is required. Install it with: pip install tensorflow")

    class_names = config.FER2013_CLASSES
    train_dir = fer_dir / "train"
    test_dir = fer_dir / "test"

    problems = verify_fer2013(fer_dir)
    if problems:
        raise ValueError("FER2013 dataset problems:\n" + "\n".join(problems))

    # Scan splits
    train_scan = scan_split(train_dir, class_names, verify=verify)
    test_scan = scan_split(test_dir, class_names, verify=verify)

    # Print class distribution
    print(f"\n{'class':<12}{'train':>8}{'test':>8}")
    for cls in class_names:
        print(f"{cls:<12}{len(train_scan[cls]):>8}{len(test_scan[cls]):>8}")
    total_train = sum(len(v) for v in train_scan.values())
    total_test = sum(len(v) for v in test_scan.values())
    print(f"{'total':<12}{total_train:>8}{total_test:>8}\n")

    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=val_fraction,
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=val_fraction,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    flow_kwargs = dict(
        target_size=(image_size, image_size),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        classes=class_names,
        seed=seed,
    )

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        subset="training",
        shuffle=True,
        **flow_kwargs,
    )
    val_gen = val_datagen.flow_from_directory(
        str(train_dir),
        subset="validation",
        shuffle=False,
        **flow_kwargs,
    )
    test_gen = test_datagen.flow_from_directory(
        str(test_dir),
        shuffle=False,
        **flow_kwargs,
    )

    return train_gen, val_gen, test_gen, class_names

