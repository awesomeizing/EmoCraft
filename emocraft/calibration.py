"""Validation-only class decision calibration for facial emotion models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


CALIBRATED_CLASSES = ("angry", "fear", "sad")


def apply_decision_bias(probabilities: np.ndarray, decision_bias: list[float] | np.ndarray | None) -> np.ndarray:
    """Apply additive log-probability offsets to reduce systematic class under-calling."""
    probs = np.asarray(probabilities, dtype=np.float32)
    if decision_bias is None:
        return probs
    bias = np.asarray(decision_bias, dtype=np.float32)
    if probs.shape[-1] != bias.size:
        return probs
    logits = np.log(np.maximum(probs, 1e-8)) + bias
    logits -= np.max(logits, axis=-1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / np.sum(adjusted, axis=-1, keepdims=True)


def tune_target_class_bias(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    target_classes: tuple[str, ...] = CALIBRATED_CLASSES,
    max_bias: float = 1.5,
    step: float = 0.05,
    passes: int = 4,
) -> list[float]:
    """Tune small per-class logit offsets using validation labels only.

    Coordinate search maximizes the mean F1 of the configured target classes.
    FER2013's validation split is used for tuning; the official test split is
    never involved in choosing these values.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.int32)
    class_names = list(class_names)
    target_ids = [class_names.index(c) for c in target_classes if c in class_names]
    if not target_ids or probs.ndim != 2 or probs.shape[1] != len(class_names):
        return [0.0] * len(class_names)

    log_probs = np.log(np.maximum(probs, 1e-8))
    bias = np.zeros(len(class_names), dtype=np.float32)
    candidates = np.arange(0.0, max_bias + step / 2, step, dtype=np.float32)

    def objective(candidate_bias: np.ndarray) -> float:
        predictions = np.argmax(log_probs + candidate_bias, axis=1)
        scores = [
            f1_score(y_true == class_id, predictions == class_id, zero_division=0)
            for class_id in target_ids
        ]
        return float(np.mean(scores))

    for _ in range(max(1, passes)):
        for class_id in target_ids:
            best_score = -1.0
            best_value = float(bias[class_id])
            for value in candidates:
                trial = bias.copy()
                trial[class_id] = value
                score = objective(trial)
                if score > best_score:
                    best_score = score
                    best_value = float(value)
            bias[class_id] = best_value
    return bias.tolist()
