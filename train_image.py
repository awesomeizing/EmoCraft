"""
Train the EmoCraft image CNN on FER2013.

Usage:
    python train_image.py [--epochs N] [--batch-size N] [--fer-dir PATH]

The trained model and metadata are saved to models/emocraft_image.keras
and models/emocraft_image_meta.json.

Dataset: FER2013 (msambare/fer2013 from Kaggle)
  - 7 classes: angry, disgust, fear, happy, neutral, sad, surprise
  - train/: 28,709 images used for training
  - 10% of train/ withheld for validation (stratified)
  - test/: 7,178 images used for one-shot evaluation only (never for training)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Keep TensorFlow's optimized CPU kernels enabled unless the user overrides it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

# Add project root to path so `import emocraft` works
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from emocraft import config
from emocraft.fer2013_data import verify_fer2013, make_keras_generators
from emocraft.model import build_emotion_cnn, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmoCraft image model on FER2013")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--fer-dir", type=Path, default=config.FER2013_DIR)
    p.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    p.add_argument("--outputs-dir", type=Path, default=config.OUTPUTS_DIR)
    p.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--resume", action="store_true",
                   help="Continue from the saved image model; --epochs is the total target")
    p.add_argument("--initial-epoch", type=int, default=None,
                   help="Override the saved completed epoch count when resuming a weights checkpoint")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip per-image file validation (faster but less safe)")
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

    # Reproducibility
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("EmoCraft — Image Model Training (FER2013)")
    print("=" * 60)
    print(f"FER2013 dir : {args.fer_dir}")
    print(f"Models dir  : {args.models_dir}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Image size  : {args.image_size}")
    print(f"Val fraction: {args.val_fraction}")
    print(f"Device      : {_device_info()}")

    # ── Validate dataset ─────────────────────────────────────────────────────
    problems = verify_fer2013(args.fer_dir)
    if problems:
        print("\nERROR: FER2013 dataset problems:")
        for p in problems:
            print(" ", p)
        print(
            "\nHow to get FER2013:\n"
            "  1. pip install kaggle\n"
            "  2. Place your kaggle.json in ~/.kaggle/\n"
            "  3. kaggle datasets download -d msambare/fer2013 -p datasets/fer2013\n"
            "  4. Unzip: datasets/fer2013/fer2013.zip\n"
            "  5. Set --fer-dir datasets/fer2013  (or EMOCRAFT_FER2013_DIR env var)"
        )
        sys.exit(1)

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("\n[data] Loading FER2013 ...")
    from emocraft.fer2013_data import load_fer2013_tensors

    train_x, train_y, val_x, val_y, test_x, test_y, class_names = load_fer2013_tensors(
        fer_dir=args.fer_dir,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Samples: train={train_x.shape[0]}  |  val={val_x.shape[0]}  |  test={test_x.shape[0]}")

    # ── Smoothed class weights (help rare classes without letting them dominate) ──
    counts = np.bincount(train_y, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    raw_weights = np.sqrt(total / (num_classes * np.maximum(counts, 1)))
    raw_weights /= np.average(raw_weights, weights=counts)
    class_weights = {i: float(raw_weights[i]) for i in range(num_classes)}
    print(f"\nClass weights: { {class_names[i]: round(v, 3) for i, v in class_weights.items()} }")

    # ── Build model ───────────────────────────────────────────────────────────
    model_path = args.models_dir / "emocraft_image.keras"
    prior_weights = None
    prior_val_accuracy = None
    initial_epoch = 0
    if args.resume:
        model = None
        if model_path.is_file():
            try:
                candidate = keras.models.load_model(str(model_path))
                candidate.get_layer("block1_conv1")
                model = candidate
            except (ValueError, KeyError):
                model = None
        if model is None:
            model = build_emotion_cnn(num_classes=num_classes, image_size=args.image_size)
            best_weights_path = args.models_dir / "emocraft_image_best.weights.h5"
            if not best_weights_path.is_file():
                raise FileNotFoundError(
                    "Could not resume: neither a compatible saved model nor "
                    f"best-weight checkpoint exists in {args.models_dir}"
                )
            model.load_weights(str(best_weights_path))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        prior_weights = model.get_weights()
        prior_val_accuracy = accuracy_score(val_y, np.argmax(
            model.predict(val_x, batch_size=args.batch_size, verbose=0), axis=1
        ))
        previous_summary = args.outputs_dir / "image_eval_summary.json"
        if previous_summary.is_file():
            previous = json.loads(previous_summary.read_text(encoding="utf-8"))
            initial_epoch = int(previous.get("epochs_run", 0))
        if args.initial_epoch is not None:
            initial_epoch = args.initial_epoch
        if args.epochs <= initial_epoch:
            raise ValueError(
                f"--epochs is the total target ({args.epochs}); it must exceed "
                f"the saved model's {initial_epoch} completed epochs."
            )
        print(f"[resume] Starting from epoch {initial_epoch}; "
              f"previous validation accuracy={prior_val_accuracy:.4f}")
    else:
        model = build_emotion_cnn(num_classes=num_classes, image_size=args.image_size)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    args.models_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = args.models_dir / "emocraft_image_best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            initial_value_threshold=prior_val_accuracy,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.EARLY_STOP_PATIENCE,
            baseline=prior_val_accuracy,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[train] Starting training for up to {args.epochs} epochs ...")
    t0 = time.time()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    best_val_accuracy = max(
        ([prior_val_accuracy] if prior_val_accuracy is not None else [])
        + history.history["val_accuracy"]
    )
    if prior_weights is not None and max(history.history["val_accuracy"]) <= prior_val_accuracy:
        # Keep the previous model if the continuation did not beat it.
        model.set_weights(prior_weights)
    elapsed = time.time() - t0
    print(f"\n[train] Training complete in {elapsed/60:.1f} min")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[eval] Evaluating on held-out test set ...")
    probs = model.predict(test_x, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_y

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nTest accuracy         : {acc:.4f}")
    print(f"Test balanced accuracy: {bal_acc:.4f}")
    print(f"Test macro-F1         : {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                digits=4, zero_division=0))

    # ── Save model + metadata ─────────────────────────────────────────────────
    weights_path, meta_path = save_model(
        model, class_names, args.image_size, args.models_dir, "emocraft_image"
    )

    # ── Save plots ────────────────────────────────────────────────────────────
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    _save_training_curves(history, args.outputs_dir / "image_training_curves.png")
    _save_confusion_matrix(y_true, y_pred, class_names,
                           args.outputs_dir / "image_confusion_matrix.png")

    # ── Save evaluation summary ───────────────────────────────────────────────
    summary = {
        "dataset": "FER2013 (msambare/fer2013, Kaggle)",
        "split": "train + 10% val withheld from train; test is official FER2013 test",
        "class_names": class_names,
        "test_accuracy": acc,
        "test_balanced_accuracy": bal_acc,
        "test_macro_f1": macro_f1,
        "per_class": json.loads(
            classification_report(y_true, y_pred,
                                  target_names=class_names,
                                  output_dict=True, zero_division=0).__str__()
            .replace("'", '"')
        ) if False else classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0
        ),
        "epochs_run": initial_epoch + len(history.history["accuracy"]),
        "epochs_this_run": len(history.history["accuracy"]),
        "best_validation_accuracy": best_val_accuracy,
        "training_seconds": elapsed,
        "model_path": str(weights_path),
        "note": "Accuracy on the FER2013 dataset. Real-world performance on new images "
                "depends on capture conditions and may differ significantly.",
    }
    summary_path = args.outputs_dir / "image_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[eval] Summary -> {summary_path}")
    print("[eval] Training curves -> ", args.outputs_dir / "image_training_curves.png")
    print("[eval] Confusion matrix -> ", args.outputs_dir / "image_confusion_matrix.png")
    print("\nDone. Launch the app with:")
    print("  streamlit run app.py")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _device_info() -> str:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return f"GPU x {len(gpus)}" if gpus else "CPU"
    except Exception:
        return "unknown"


def _save_training_curves(history, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_confusion_matrix(y_true, y_pred, class_names: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    ax.set_title("FER2013 test set confusion matrix")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

