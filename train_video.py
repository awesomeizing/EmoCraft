"""
Train EmoCraft video model on CREMA-D (or fall back to RAVDESS demo).

Usage:
    python train_video.py [--cremad-dir PATH] [--ravdess-dir PATH] [--epochs N]

CREMA-D is strongly preferred:
  - 91 actors × 6 emotions → actor-disjoint split (no actor in both train and test)
  - 7,442 clips total

RAVDESS fallback (pilot / smoke-test only):
  - Only actors 05 and 06 currently downloaded → NOT a reliable generalization estimate
  - Results are clearly labeled as a demo, not production metrics.

Saved artefacts:
  models/emocraft_video.keras            model weights
  models/emocraft_video_meta.json        class list + config
  outputs/video_training_curves.png
  outputs/video_confusion_matrix.png
  outputs/video_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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

from emocraft import config
from emocraft.cremad_data import verify_cremad, scan_cremad, actor_disjoint_split, extract_face_crops
from emocraft.model import build_emotion_cnn, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmoCraft video model")
    p.add_argument("--cremad-dir", type=Path, default=config.CREMAD_DIR)
    p.add_argument("--ravdess-dir", type=Path, default=config.RAVDESS_DIR)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    p.add_argument("--outputs-dir", type=Path, default=config.OUTPUTS_DIR)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--frames-per-clip", type=int, default=8)
    p.add_argument("--face-model", type=Path, default=config.FACE_DETECTOR_MODEL)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import keras
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score, ConfusionMatrixDisplay
    )

    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("EmoCraft — Video Model Training")
    print("=" * 60)

    # ── Try CREMA-D ───────────────────────────────────────────────────────────
    cremad_ok, cremad_problems = verify_cremad(args.cremad_dir)

    if cremad_ok:
        print(f"[data] Using CREMA-D from {args.cremad_dir}")
        records = scan_cremad(args.cremad_dir)
        if not records:
            cremad_ok = False
            cremad_problems.append("scan_cremad returned zero records despite verify passing")

    if cremad_ok:
        class_names = config.CREMAD_CLASSES
        print(f"[data] Clips found: {len(records)}")
        print(f"[data] Actors: {len({r['actor'] for r in records})}")
        print(f"[data] Classes: {dict(sorted(Counter(r['label'] for r in records).items()))}")

        train_recs, val_recs, test_recs = actor_disjoint_split(records, seed=args.seed)
        print(f"[split] train clips={len(train_recs)} val clips={len(val_recs)} test clips={len(test_recs)}")

        print("[data] Extracting face crops from video frames (this may take a while)...")
        train_recs, train_fail = extract_face_crops(train_recs, args.face_model,
                                                     args.image_size, args.frames_per_clip)
        val_recs, val_fail = extract_face_crops(val_recs, args.face_model,
                                                  args.image_size, args.frames_per_clip)
        test_recs, test_fail = extract_face_crops(test_recs, args.face_model,
                                                    args.image_size, args.frames_per_clip)
        print(f"[data] Usable: train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")
        print(f"[data] Failed (no face): train={len(train_fail)} val={len(val_fail)} test={len(test_fail)}")
        dataset_note = (
            "CREMA-D; actor-disjoint split (15% actors held out as test, "
            "10% as validation). All 91 actors used."
        )

    else:
        # ── Fall back to RAVDESS (pilot/demo only) ────────────────────────────
        print("\n[data] CREMA-D not available. Issues:")
        for pr in cremad_problems:
            print(" ", pr)
        print("\n[data] Falling back to RAVDESS pilot (actors 05 and 06).")
        print("[WARN] RAVDESS pilot: only 2 actors - results are NOT reliable estimates of generalization!")
        print("       Use this only as a smoke-test / demo. Obtain CREMA-D for real video training.\n")

        from emocraft.ravdess_data import load_ravdess_records, extract_face_crops as ravdess_crops, split_by_actor
        records = load_ravdess_records(args.ravdess_dir)
        if not records:
            print("[ERROR] No RAVDESS clips found either. Cannot train video model.")
            print("        Please supply at least CREMA-D or RAVDESS data.")
            sys.exit(1)

        class_names = config.FER2013_CLASSES  # RAVDESS has 8 classes mapped to 7 FER labels
        class_names = [c for c in class_names if c != "surprise"]  # RAVDESS has no surprise in speech-video
        # We'll use the labels actually present in RAVDESS
        present_labels = sorted({r["label"] for r in records})
        class_names = present_labels
        print(f"[data] RAVDESS clips: {len(records)}, labels: {Counter(r['label'] for r in records)}")

        train_recs, val_recs, test_recs, train_actor, test_actor = split_by_actor(records, seed=args.seed)

        train_recs, _ = ravdess_crops(train_recs, args.face_model, args.image_size, args.frames_per_clip)
        val_recs, _ = ravdess_crops(val_recs, args.face_model, args.image_size, args.frames_per_clip)
        test_recs, _ = ravdess_crops(test_recs, args.face_model, args.image_size, args.frames_per_clip)
        dataset_note = (
            f"RAVDESS pilot (actors {train_actor} train/val, {test_actor} test). "
            "TWO ACTORS ONLY — not a reliable generalization estimate."
        )

    # ── Build tensors ─────────────────────────────────────────────────────────
    cls_to_id = {c: i for i, c in enumerate(class_names)}

    def to_tensors(recs):
        xs, ys = [], []
        for r in recs:
            for crop in r["crops"]:
                xs.append(crop)
                ys.append(cls_to_id[r["label"]])
        if not xs:
            return np.zeros((0, args.image_size, args.image_size, 1), np.float32), np.zeros(0, np.int32)
        x = np.stack(xs)[:, :, :, np.newaxis]
        y = np.array(ys, np.int32)
        return x, y

    train_x, train_y = to_tensors(train_recs)
    val_x, val_y = to_tensors(val_recs)
    test_x, test_y = to_tensors(test_recs)

    print(f"\n[tensors] train={train_x.shape} val={val_x.shape} test={test_x.shape}")
    if train_x.shape[0] == 0:
        print("[ERROR] No training samples. Aborting.")
        sys.exit(1)

    # ── Build & train model ────────────────────────────────────────────────────
    num_classes = len(class_names)
    model = build_emotion_cnn(num_classes=num_classes, image_size=args.image_size)

    # Class weights
    counts = np.bincount(train_y, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    class_weights = {i: float(total / (num_classes * max(counts[i], 1))) for i in range(num_classes)}

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.models_dir / "emocraft_video_best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                       patience=config.EARLY_STOP_PATIENCE,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=config.LR_REDUCE_PATIENCE,
                                          min_lr=1e-6, verbose=1),
    ]

    print(f"\n[train] Training for up to {args.epochs} epochs ...")
    t0 = time.time()
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"[train] Done in {elapsed/60:.1f} min")

    # ── Evaluate: frame-level then clip-level ──────────────────────────────────
    print("\n[eval] Frame-level evaluation on test frames ...")
    test_probs = model.predict(test_x, verbose=1)
    frame_preds = np.argmax(test_probs, axis=1)

    frame_acc = accuracy_score(test_y, frame_preds)
    frame_bal = balanced_accuracy_score(test_y, frame_preds)
    frame_f1 = f1_score(test_y, frame_preds, average="macro", zero_division=0)
    print(f"Frame accuracy: {frame_acc:.4f}  balanced: {frame_bal:.4f}  macro-F1: {frame_f1:.4f}")

    # Clip-level majority vote
    print("\n[eval] Clip-level majority-vote evaluation ...")
    y_true_clip, y_pred_clip = [], []
    frame_ptr = 0
    for rec in test_recs:
        n = len(rec["crops"])
        if n == 0:
            continue
        clip_preds = frame_preds[frame_ptr:frame_ptr + n]
        frame_ptr += n
        vote = Counter(clip_preds.tolist()).most_common(1)[0][0]
        y_true_clip.append(cls_to_id[rec["label"]])
        y_pred_clip.append(vote)

    clip_acc = accuracy_score(y_true_clip, y_pred_clip) if y_true_clip else 0.0
    clip_bal = balanced_accuracy_score(y_true_clip, y_pred_clip) if y_true_clip else 0.0
    clip_f1 = f1_score(y_true_clip, y_pred_clip, average="macro", zero_division=0) if y_true_clip else 0.0
    print(f"Clip accuracy : {clip_acc:.4f}  balanced: {clip_bal:.4f}  macro-F1: {clip_f1:.4f}")
    print(f"Clips evaluated: {len(y_true_clip)}")
    if y_true_clip:
        print(classification_report(y_true_clip, y_pred_clip,
                                    target_names=class_names, digits=4, zero_division=0))

    # ── Save ──────────────────────────────────────────────────────────────────
    weights_path, meta_path = save_model(model, class_names, args.image_size,
                                          args.models_dir, "emocraft_video")

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    _save_curves(history, args.outputs_dir / "video_training_curves.png")
    if y_true_clip:
        _save_cm(y_true_clip, y_pred_clip, class_names,
                 args.outputs_dir / "video_confusion_matrix.png")

    summary = {
        "dataset": dataset_note,
        "class_names": class_names,
        "frame_accuracy": frame_acc,
        "frame_balanced_accuracy": frame_bal,
        "frame_macro_f1": frame_f1,
        "clip_accuracy": clip_acc,
        "clip_balanced_accuracy": clip_bal,
        "clip_macro_f1": clip_f1,
        "clips_evaluated": len(y_true_clip),
        "epochs_run": len(history.history["accuracy"]),
        "training_seconds": elapsed,
        "model_path": str(weights_path),
        "warning": (
            "RAVDESS pilot: two-actor results must NOT be interpreted as reliable generalization estimates."
            if "RAVDESS" in dataset_note else
            "Metrics are on actor-held-out test clips from CREMA-D."
        ),
    }
    (args.outputs_dir / "video_eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n[done] Artifacts saved in", args.outputs_dir)
    print("       Launch the app: streamlit run app.py")


def _save_curves(history, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history.history["accuracy"], label="train")
    a1.plot(history.history["val_accuracy"], label="val")
    a1.set_title("Accuracy"); a1.legend()
    a2.plot(history.history["loss"], label="train")
    a2.plot(history.history["val_loss"], label="val")
    a2.set_title("Loss"); a2.legend()
    fig.tight_layout(); fig.savefig(str(out), dpi=150); plt.close(fig)


def _save_cm(y_true, y_pred, class_names, out: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("Video test-set confusion matrix (clip-level)")
    fig.tight_layout(); fig.savefig(str(out), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

