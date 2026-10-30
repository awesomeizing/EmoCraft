"""Small, actor-disjoint EmoCraft proof-of-run on the locally downloaded RAVDESS videos.

This deliberately uses only RAVDESS video-only speech clips for which the emotion
is in the seven-class FER label set. Actor 05 is split into train/validation clips;
actor 06 is held out for the reported test score. This is a runnable demo, not a
reliable estimate of real-world emotion recognition performance.
"""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
RAVDESS_DIR = Path(__file__).resolve().parent / "RAVDESS"
OUTPUT_DIR = RAVDESS_DIR / "run_output"
EXTRACT_DIR = OUTPUT_DIR / "video_only_speech"
IMAGE_SIZE = 48
FRAMES_PER_CLIP = 5
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

EMOTION_CODES = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_ravdess_name(name: str):
    fields = Path(name).stem.split("-")
    if len(fields) != 7:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = fields
    if modality != "02" or channel != "01":
        return None  # only video-only speech; exclude audio-video duplicates and songs
    label = EMOTION_CODES.get(emotion)  # omit RAVDESS calm because FER has no calm class
    if label is None:
        return None
    return {"label": label, "actor": actor, "emotion_id": emotion}


def extract_ravdess_clips():
    archives = sorted(RAVDESS_DIR.glob("Video_Speech_Actor_*.zip"))
    if not archives:
        raise FileNotFoundError(f"No Video_Speech_Actor_*.zip files found in {RAVDESS_DIR}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() != ".mp4":
                    continue
                parsed = parse_ravdess_name(member.filename)
                if parsed is None:
                    continue
                actor_dir = EXTRACT_DIR / f"Actor_{parsed['actor']}"
                actor_dir.mkdir(parents=True, exist_ok=True)
                destination = actor_dir / Path(member.filename).name
                if not destination.is_file() or destination.stat().st_size != member.file_size:
                    with zf.open(member, "r") as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                if destination.stat().st_size != member.file_size:
                    raise IOError(f"Incomplete extraction: {destination}")
                records.append({**parsed, "path": destination, "filename": destination.name})
    records.sort(key=lambda r: (r["actor"], r["label"], r["filename"]))
    if not records:
        raise RuntimeError("No compatible video-only speech clips were found in the supplied archives.")
    return records


def extract_face_frames(records):
    # RAVDESS uses a fixed, centered camera setup. This conservative central
    # crop keeps the face/head region and avoids relying on OpenCV's removed
    # Haar Cascade API in OpenCV 5. It is specific to this posed dataset.
    usable, face_failures = [], []
    for record in records:
        cap = cv2.VideoCapture(str(record["path"]))
        if not cap.isOpened():
            face_failures.append({"file": record["filename"], "reason": "video could not be opened"})
            continue
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 1:
                face_failures.append({"file": record["filename"], "reason": "no frames reported"})
                continue
            indices = np.unique(np.linspace(0, frame_count - 1, min(FRAMES_PER_CLIP, frame_count)).round().astype(int))
            crops, previews = [], []
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                x1, x2 = int(0.30 * w), int(0.70 * w)
                y1, y2 = int(0.03 * h), int(0.78 * h)
                crop = gray[y1:y2, x1:x2]
                crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                crops.append(crop)
                if not previews:
                    preview = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    previews.append((preview, (x1, y1, x2 - x1, y2 - y1)))
            if not crops:
                face_failures.append({"file": record["filename"], "reason": "no sampled frames could be decoded"})
                continue
            usable.append({**record, "frames": crops, "preview": previews[0]})
        finally:
            cap.release()
    return usable, face_failures


def split_records_by_actor(records):
    actors = sorted({r["actor"] for r in records})
    if len(actors) < 2:
        raise RuntimeError("At least two actors are needed for an actor-held-out test split.")
    train_actor, test_actor = actors[0], actors[-1]
    trainval = [r for r in records if r["actor"] == train_actor]
    test = [r for r in records if r["actor"] == test_actor]
    if len(actors) > 2:
        print(f"Note: using actors {train_actor} and {test_actor}; other supplied actors are not used in this small demo.")
    labels = [r["label"] for r in trainval]
    train_records, val_records = train_test_split(
        trainval, test_size=0.20, random_state=SEED, stratify=labels
    )
    return train_records, val_records, test, train_actor, test_actor


def expand_frames(records):
    images, labels, source_clips = [], [], []
    for record in records:
        images.extend(record["frames"])
        labels.extend([CLASS_TO_ID[record["label"]]] * len(record["frames"]))
        source_clips.extend([record["filename"]] * len(record["frames"]))
    x = np.asarray(images, dtype=np.float32)[:, None, :, :] / 255.0
    y = np.asarray(labels, dtype=np.int64)
    return x, y, source_clips


class SmallEmotionCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loader(x, y, shuffle):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train_model(train_x, train_y, val_x, val_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallEmotionCNN(len(CLASS_NAMES)).to(device)
    train_loader, val_loader = make_loader(train_x, train_y, True), make_loader(val_x, val_y, False)
    counts = np.bincount(train_y, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = len(train_y) / (len(CLASS_NAMES) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_loss, best_state = float("inf"), None

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = correct = seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if random.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            seen += len(yb)

        model.eval()
        val_loss_sum = val_correct = val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += float(criterion(logits, yb).item()) * len(yb)
                val_correct += int((logits.argmax(1) == yb).sum().item())
                val_seen += len(yb)

        history["train_loss"].append(train_loss_sum / max(seen, 1))
        history["val_loss"].append(val_loss_sum / max(val_seen, 1))
        history["train_accuracy"].append(correct / max(seen, 1))
        history["val_accuracy"].append(val_correct / max(val_seen, 1))
        if history["val_loss"][-1] < best_val_loss:
            best_val_loss = history["val_loss"][-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"train_loss={history['train_loss'][-1]:.3f} train_acc={history['train_accuracy'][-1]:.3f} "
            f"val_loss={history['val_loss'][-1]:.3f} val_acc={history['val_accuracy'][-1]:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, device


def predict_clip(model, device, record):
    x = np.asarray(record["frames"], dtype=np.float32)[:, None, :, :] / 255.0
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        frame_predictions = logits.argmax(dim=1).cpu().numpy()
    counts = Counter(int(i) for i in frame_predictions)
    pred_id = counts.most_common(1)[0][0]
    return CLASS_NAMES[pred_id]


def save_plots(history, y_true, y_pred):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(history["train_accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Frame accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150); plt.close(fig)

    labels_present = [name for name in CLASS_NAMES if name in y_true]
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels_present).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("RAVDESS held-out actor: clip-level confusion matrix")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)


def main():
    set_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    print("RAVDESS directory:", RAVDESS_DIR)
    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

    clips = extract_ravdess_clips()
    print("Compatible clips:", len(clips), "| by actor:", dict(sorted(Counter(r["actor"] for r in clips).items())))
    print("Clip labels:", dict(sorted(Counter(r["label"] for r in clips).items())))
    usable, failures = extract_face_frames(clips)
    print(f"Clips with sampled centered crops: {len(usable)} / {len(clips)}; skipped: {len(failures)}")
    if failures:
        print("First skipped clip:", failures[0])

    train_records, val_records, test_records, train_actor, test_actor = split_records_by_actor(usable)
    train_x, train_y, _ = expand_frames(train_records)
    val_x, val_y, _ = expand_frames(val_records)
    test_x, test_y, _ = expand_frames(test_records)
    print(f"Split by actor: train/validation actor {train_actor}; held-out test actor {test_actor}")
    print(f"Face crops: train={len(train_y)} val={len(val_y)} test={len(test_y)}")
    print("Training frame counts:", {CLASS_NAMES[i]: int(n) for i, n in enumerate(np.bincount(train_y, minlength=len(CLASS_NAMES)))})

    model, history, device = train_model(train_x, train_y, val_x, val_y)
    y_true, y_pred = [], []
    for record in test_records:
        y_true.append(record["label"])
        y_pred.append(predict_clip(model, device, record))

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    print("\nHeld-out actor-level test results (pilot only)")
    print(f"Video clips: {len(y_true)} | accuracy={acc:.3f} | balanced_accuracy={bal_acc:.3f} | macro_F1={macro_f1:.3f}")
    print(classification_report(y_true, y_pred, labels=CLASS_NAMES, target_names=CLASS_NAMES, digits=3, zero_division=0))

    save_plots(history, y_true, y_pred)
    model_path = OUTPUT_DIR / "ravdess_demo_cnn.pth"
    torch.save({"state_dict": model.state_dict(), "class_names": CLASS_NAMES, "image_size": IMAGE_SIZE}, model_path)

    preview_record = next((r for r in test_records if r["label"] == "happy"), test_records[0])
    preview_pred = predict_clip(model, device, preview_record)
    rgb, (x, y, w, h) = preview_record["preview"]
    preview = rgb.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(preview, f"true={preview_record['label']} pred={preview_pred}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    preview_path = OUTPUT_DIR / "sample_prediction.png"
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    summary = {
        "dataset": "RAVDESS video-only speech; actors currently downloaded are 05 and 06",
        "train_actor": train_actor,
        "held_out_test_actor": test_actor,
        "compatible_clips": len(clips),
        "clips_with_face_frames": len(usable),
        "skipped_clips": failures,
        "class_names": CLASS_NAMES,
        "frame_crops": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "clip_accuracy": acc,
        "clip_balanced_accuracy": bal_acc,
        "clip_macro_f1": macro_f1,
        "preview": {"filename": preview_record["filename"], "true": preview_record["label"], "predicted": preview_pred},
        "warning": "Two actors only; this is a runnable demonstration, not a generalization claim.",
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved artifacts:")
    for path in (model_path, OUTPUT_DIR / "training_curves.png", OUTPUT_DIR / "confusion_matrix.png",
                 preview_path, OUTPUT_DIR / "run_summary.json"):
        print(" ", path)
    print("\nLimit: only two actors were downloaded; results are small-sample and should not be presented as production accuracy.")


if __name__ == "__main__":
    main()
