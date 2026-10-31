"""HSEmotion ONNX inference and the validation-tuned image ensemble."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from emocraft.calibration import apply_decision_bias


class HSEmotionONNX:
    """AFEW EfficientNet-B0 facial expression model exported to ONNX."""

    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    input_size = 224
    _class_indices = np.array([0, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a grayscale or BGR face image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Cannot predict an empty face image")
        rgb = self._to_rgb(image)
        # Remove a small border of detector padding before resizing. This was
        # also used when the image ensemble weight was selected on validation.
        height, width = rgb.shape[:2]
        margin_x, margin_y = int(0.08 * width), int(0.08 * height)
        if margin_x or margin_y:
            rgb = rgb[margin_y:height - margin_y or height,
                      margin_x:width - margin_x or width]
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        return (rgb - self._mean) / self._std

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        scores = np.exp(logits)
        return scores / np.sum(scores, axis=-1, keepdims=True)

    def predict_faces(self, face_images: list[np.ndarray]) -> np.ndarray:
        """Predict a list of grayscale or BGR face crops; returns seven FER scores."""
        if not face_images:
            return np.empty((0, len(self.class_names)), dtype=np.float32)
        prepared = np.stack([self._prepare(image) for image in face_images])
        blob = np.transpose(prepared, (0, 3, 1, 2)).astype(np.float32, copy=False)
        self.net.setInput(blob, "input")
        logits = np.asarray(self.net.forward("output"), dtype=np.float32)
        logits = logits[:, self._class_indices]
        return self._softmax(logits)

    def predict_face(self, bgr: np.ndarray) -> np.ndarray:
        """Return seven class probabilities for one BGR or grayscale face crop."""
        return self.predict_faces([bgr])[0]

    def predict_gray_batch(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Predict FER-style normalized grayscale crops in batches."""
        values = np.asarray(images)
        if values.ndim == 4 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim != 3:
            raise ValueError("Expected images with shape (batch, height, width[, 1])")

        results = []
        for start in range(0, len(values), batch_size):
            chunk = values[start:start + batch_size]
            faces = [
                np.clip(image * 255.0 if np.issubdtype(image.dtype, np.floating) else image,
                        0, 255).astype(np.uint8)
                for image in chunk
            ]
            results.append(self.predict_faces(faces))
        return np.concatenate(results, axis=0) if results else np.empty((0, 7), np.float32)


class ImageHSEEnsemble:
    """Blend calibrated FER2013 CNN scores with HSEmotion face scores."""

    def __init__(self, keras_model, hse_model: HSEmotionONNX, hse_weight: float = 0.4):
        self.keras_model = keras_model
        self.hse_model = hse_model
        self.hse_weight = float(np.clip(hse_weight, 0.0, 1.0))

    def predict_face(
        self,
        face_bgr: np.ndarray,
        crop_f32: np.ndarray,
        decision_bias: list[float] | None = None,
    ) -> np.ndarray:
        crop = np.asarray(crop_f32, dtype=np.float32)
        variants = np.stack((crop, np.fliplr(crop)), axis=0)[..., np.newaxis]
        keras_probs = np.asarray(self.keras_model.predict(variants, verbose=0), dtype=np.float32).mean(axis=0)
        hse_probs = self.hse_model.predict_face(face_bgr)
        blended = (1.0 - self.hse_weight) * keras_probs + self.hse_weight * hse_probs
        return apply_decision_bias(blended, decision_bias).astype(np.float32)


def tune_hse_ensemble_weight(
    keras_probabilities: np.ndarray,
    hse_probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    target_classes: tuple[str, ...] = ("angry", "fear", "sad"),
    candidates: np.ndarray | None = None,
) -> float:
    """Select an HSE blend weight on validation data, balancing global and target F1."""
    from sklearn.metrics import f1_score

    keras_probs = np.asarray(keras_probabilities, dtype=np.float32)
    hse_probs = np.asarray(hse_probabilities, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.int32)
    if keras_probs.shape != hse_probs.shape or keras_probs.ndim != 2:
        return 0.0
    if keras_probs.shape[0] != y_true.shape[0] or keras_probs.shape[1] != len(class_names):
        return 0.0

    target_ids = [class_names.index(name) for name in target_classes if name in class_names]
    candidate_weights = candidates if candidates is not None else np.arange(0.0, 0.801, 0.05)
    best_weight, best_score = 0.0, -1.0
    for weight in candidate_weights:
        blended = (1.0 - float(weight)) * keras_probs + float(weight) * hse_probs
        predicted = np.argmax(blended, axis=1)
        overall_f1 = f1_score(y_true, predicted, average="macro", zero_division=0)
        target_f1 = np.mean([
            f1_score(y_true == class_id, predicted == class_id, zero_division=0)
            for class_id in target_ids
        ]) if target_ids else overall_f1
        score = 0.75 * overall_f1 + 0.25 * target_f1
        if score > best_score:
            best_score, best_weight = float(score), float(weight)
    return best_weight
