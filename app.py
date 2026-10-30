"""
EmoCraft Streamlit Application
================================
Facial emotion recognition from uploaded images and videos.

Launch:
    streamlit run app.py

Environment variables (optional):
    EMOCRAFT_FER2013_DIR    Path to FER2013 dataset root (train/ + test/)
    EMOCRAFT_MODELS_DIR     Path to saved model directory
    EMOCRAFT_FACE_MODEL     Path to YuNet ONNX model file
    EMOCRAFT_IMAGE_SIZE     Input image size (default: 48)
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Project path setup ────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from emocraft import config
from emocraft.model import load_model
from emocraft.inference import predict_image, predict_video, write_annotated_video


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoCraft — Emotion Recognition",
    page_icon="🎭",
    layout="wide",
)

# ── Emotion display colours ───────────────────────────────────────────────────
EMOTION_COLORS = {
    "angry":    "#e53935",
    "disgust":  "#8bc34a",
    "fear":     "#7b1fa2",
    "happy":    "#ffb300",
    "neutral":  "#546e7a",
    "sad":      "#1565c0",
    "surprise": "#f4511e",
}

EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}


# ── Cached model loader ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading emotion model...")
def load_image_model():
    try:
        model, meta = load_model(config.MODELS_DIR, "emocraft_image")
        return model, meta, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Failed to load image model: {e}"


@st.cache_resource(show_spinner="Loading video emotion model...")
def load_video_model():
    try:
        model, meta = load_model(config.MODELS_DIR, "emocraft_video")
        return model, meta, None
    except FileNotFoundError:
        return None, None, None  # Not trained yet — silently skip
    except Exception as e:
        return None, None, f"Failed to load video model: {e}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def bgr_to_rgb_bytes(bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", rgb)
    return buf.tobytes() if ok else b""


def display_confidence_bars(all_scores: dict[str, float]) -> None:
    """Render a coloured confidence bar chart for each emotion."""
    sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_emotions:
        colour = EMOTION_COLORS.get(emotion, "#90a4ae")
        emoji = EMOTION_EMOJI.get(emotion, "")
        pct = score * 100
        st.markdown(
            f"""
            <div style="margin-bottom:4px">
              <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span>{emoji} <b>{emotion}</b></span>
                <span>{pct:.1f}%</span>
              </div>
              <div style="background:#e0e0e0;border-radius:4px;height:10px;overflow:hidden">
                <div style="width:{pct:.1f}%;background:{colour};height:100%;border-radius:4px"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def result_badge(label: str, confidence: float) -> None:
    colour = EMOTION_COLORS.get(label, "#546e7a")
    emoji = EMOTION_EMOJI.get(label, "")
    st.markdown(
        f"""
        <div style="text-align:center;padding:16px;border-radius:12px;
                    background:{colour}22;border:2px solid {colour}">
          <div style="font-size:3rem">{emoji}</div>
          <div style="font-size:1.8rem;font-weight:bold;color:{colour}">{label.upper()}</div>
          <div style="font-size:1rem;color:#555">{confidence:.0%} confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.title("🎭 EmoCraft")
        st.caption("Facial emotion recognition")
        st.divider()

        st.subheader("Model Status")
        img_model, img_meta, img_err = load_image_model()
        vid_model, vid_meta, vid_err = load_video_model()

        if img_model is not None:
            st.success(f"✅ Image model loaded\n\nClasses: {', '.join(img_meta['class_names'])}")
        elif img_err:
            st.error(f"❌ Image model: {img_err}")
            st.info("Run `python train_image.py` to train the image model first.")
        else:
            st.warning("⚠ Image model not trained yet.")

        if vid_model is not None:
            st.success(f"✅ Video model loaded\n\nClasses: {', '.join(vid_meta['class_names'])}")
        elif vid_err:
            st.warning(f"⚠ Video model: {vid_err}")
        else:
            st.info("ℹ Video model not trained.\nRun `python train_video.py`.")

        st.divider()
        st.subheader("Face Detector")
        face_path = config.FACE_DETECTOR_MODEL
        if face_path.exists():
            st.success(f"✅ YuNet ONNX\n({face_path.stat().st_size:,} bytes)")
        else:
            st.error(f"❌ YuNet model not found:\n{face_path}")

        st.divider()
        st.caption(
            "**Dataset**: FER2013 (train) · CREMA-D or RAVDESS (video)  \n"
            "**License**: FER2013: research use; CREMA-D: ODbL; RAVDESS: CC BY-NC-SA  \n"
            "**Note**: Accuracy reflects held-out test data from the training dataset. "
            "Real-world performance may differ."
        )


# ── Image tab ─────────────────────────────────────────────────────────────────

def render_image_tab() -> None:
    st.header("📷 Image Emotion Recognition")

    img_model, img_meta, img_err = load_image_model()
    face_path = config.FACE_DETECTOR_MODEL

    if img_model is None:
        st.error(
            "Image model not loaded. Train it first:\n```\npython train_image.py\n```"
        )
        if img_err:
            st.code(img_err)
        return

    if not face_path.exists():
        st.error(
            f"YuNet face detector model not found at {face_path}.\n"
            "Run `python smoke_test.py` to download it automatically."
        )
        return

    uploaded = st.file_uploader(
        "Upload a face image (JPEG / PNG / BMP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    image_bytes = uploaded.read()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input image")
        st.image(image_bytes, use_container_width=True)

    with st.spinner("Detecting face and predicting emotion..."):
        result = predict_image(image_bytes, img_model, img_meta, face_path)

    with col2:
        st.subheader("Result")
        if result["error"]:
            st.error(result["error"])
            if result.get("annotated_bgr") is not None:
                st.image(bgr_to_rgb_bytes(result["annotated_bgr"]), use_container_width=True)
            return

        result_badge(result["label"], result["confidence"])
        st.divider()
        if not result["face_found"]:
            st.caption("ℹ Image treated as pre-cropped face (no bounding box detected).")

        st.subheader("Confidence breakdown")
        display_confidence_bars(result["all_scores"])

    if result["annotated_bgr"] is not None:
        st.divider()
        st.subheader("Annotated image")
        st.image(bgr_to_rgb_bytes(result["annotated_bgr"]), use_container_width=True)


# ── Video tab ─────────────────────────────────────────────────────────────────

def render_video_tab() -> None:
    st.header("🎬 Video Emotion Recognition")

    # Prefer video-specific model; fall back to image model
    vid_model, vid_meta, vid_err = load_video_model()
    img_model, img_meta, img_err = load_image_model()

    if vid_model is not None:
        active_model, active_meta, model_label = vid_model, vid_meta, "video model"
    elif img_model is not None:
        active_model, active_meta, model_label = img_model, img_meta, "image model (used as fallback)"
        st.warning(
            "⚠ No video-specific model found. Using the image model as a fallback. "
            "Train a dedicated video model with `python train_video.py` for better performance."
        )
    else:
        st.error("No model available. Train at least the image model: `python train_image.py`")
        return

    face_path = config.FACE_DETECTOR_MODEL
    if not face_path.exists():
        st.error(f"YuNet face detector model not found at {face_path}.")
        return

    st.caption(f"Using: **{model_label}** · Classes: {', '.join(active_meta['class_names'])}")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        target_fps = st.slider("Frames to sample per second", 0.5, 5.0, 2.0, 0.5,
                                help="Higher = more frames processed (slower but more stable)")
    with col_b:
        max_frames = st.slider("Maximum frames to process", 10, 120, 60,
                                help="Cap on the total number of sampled frames")

    uploaded = st.file_uploader(
        "Upload a video (MP4 / AVI / MOV / MKV)",
        type=["mp4", "avi", "mov", "mkv", "flv"],
        key="video_upload",
    )

    if uploaded is None:
        st.info("Upload a video to get started.")
        return

    # Save to temp file (OpenCV needs a file path)
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded.read())
        tmp_path = Path(tf.name)

    try:
        with st.spinner("Processing video — sampling frames and detecting emotions..."):
            t0 = time.time()
            result = predict_video(
                tmp_path, active_model, active_meta, face_path,
                target_fps=target_fps, max_frames=max_frames,
            )
            elapsed = time.time() - t0

        if result["error"]:
            st.error(result["error"])
            return

        # Summary
        st.success(
            f"Processed {result['frames_processed']} frames "
            f"({result['frames_with_face']} with detected face) in {elapsed:.1f}s"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Overall prediction")
            result_badge(result["label"], result["confidence"])
            st.divider()
            st.subheader("Mean confidence per emotion")
            display_confidence_bars(result["all_scores"])

        with col2:
            st.subheader("Per-frame results")
            if result["per_frame"]:
                frame_data = [
                    {"Frame": r["frame_idx"], "Emotion": r["label"],
                     "Confidence": f"{r['confidence']:.0%}"}
                    for r in result["per_frame"]
                ]
                st.dataframe(frame_data, use_container_width=True, height=300)
            else:
                st.info("No frames with detected faces.")

        # Annotated video download
        st.divider()
        st.subheader("Download annotated video")
        if result["annotated_frames_bgr"]:
            out_path = tmp_path.parent / f"annotated_{tmp_path.stem}.mp4"
            written = write_annotated_video(result["annotated_frames_bgr"], out_path, fps=target_fps)
            if written and written.exists():
                with open(written, "rb") as f:
                    st.download_button(
                        "⬇ Download annotated MP4",
                        data=f.read(),
                        file_name=f"emocraft_annotated_{uploaded.name}",
                        mime="video/mp4",
                    )
                # Show a few annotated frames
                st.subheader("Sample annotated frames")
                sample_frames = result["annotated_frames_bgr"][::max(1, len(result["annotated_frames_bgr"]) // 6)][:6]
                cols = st.columns(min(3, len(sample_frames)))
                for i, frame in enumerate(sample_frames[:3]):
                    with cols[i % 3]:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.warning("Could not write annotated video file.")
        else:
            st.info("No annotated frames to export.")

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── About tab ─────────────────────────────────────────────────────────────────

def render_about_tab() -> None:
    st.header("ℹ About EmoCraft")
    st.markdown("""
    **EmoCraft** is a facial emotion recognition system that uses a convolutional
    neural network trained on the FER2013 dataset.

    ### Architecture
    - Custom lightweight CNN (5 conv layers + global average pooling)
    - Input: 48×48 grayscale face crops
    - Face detection: OpenCV YuNet (FaceDetectorYN) — no Haar cascades needed
    - Training: TensorFlow/Keras with class-weighted loss, early stopping, LR scheduling

    ### Datasets & Licences
    | Dataset | Purpose | Classes | Licence |
    |---------|---------|---------|---------|
    | FER2013 | Image model training | 7 (angry, disgust, fear, happy, neutral, sad, surprise) | Research use (Kaggle competition) |
    | CREMA-D | Video model training (preferred) | 6 (no surprise) | Open Database Licence (ODbL) v1.0 |
    | RAVDESS | Pilot/demo fallback (2 actors only) | 7 | CC BY-NC-SA 4.0 |

    **FER2013** citation:
    > Goodfellow, I. J., et al. (2013). Challenges in Representation Learning: A report on
    > three machine learning contests. ICONIP 2013.

    **CREMA-D** citation:
    > Cao, H., Cooper, D. G., Kuchera-Morin, J., et al. (2014).
    > CREMA-D: Crowd-sourced emotional multimodal actors dataset.
    > IEEE Transactions on Affective Computing.

    **RAVDESS** citation:
    > Livingstone, S. R. & Russo, F. A. (2018).
    > The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
    > PLoS ONE 13(5): e0196391.

    ### Label alignment
    - FER2013 has 7 classes. CREMA-D has 6 (no *surprise*).
    - Models trained on different datasets have different class lists; each model's
      class list is saved with it and loaded at inference time.
    - No silent remapping: the app always displays which model is active.

    ### Honest limitations
    - FER2013 test accuracy is typically **60–68%** for CNN baselines (benchmark is hard).
    - The RAVDESS 2-actor pilot is **not** production accuracy — it is a smoke test only.
    - Performance on real-world images/videos can differ significantly from benchmark results.
    """)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    st.title("🎭 EmoCraft — Facial Emotion Recognition")
    tab_img, tab_vid, tab_about = st.tabs(["📷 Image", "🎬 Video", "ℹ About"])

    with tab_img:
        render_image_tab()

    with tab_vid:
        render_video_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()

