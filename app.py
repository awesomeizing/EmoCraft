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
    page_icon="favicon.png" if (_HERE / "favicon.png").exists() else ":material/face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# Palette: ivory / near-black / sage / terracotta / stone border / muted grey
_CSS = """
<style>
/* ── Reset & global ── */
:root {
  --bg:      #F5F3EE;
  --surface: #EDEAE3;
  --text:    #272923;
  --muted:   #6B6960;
  --sage:    #667568;
  --terra:   #AD7059;
  --border:  #D9D5CC;
  --radius:  6px;
  --font:    -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}

/* Main background */
.stApp, [data-testid="stAppViewContainer"] > .main {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font);
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid var(--border);
  background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  font-weight: 500;
  padding: 10px 20px !important;
  transition: color .15s ease, border-color .15s ease;
}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
  color: var(--terra) !important;
  border-bottom: 2px solid var(--terra) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--terra) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:focus-visible {
  outline: 2px solid var(--terra);
  outline-offset: 2px;
}

/* Buttons */
.stButton > button {
  background: var(--terra) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-weight: 500;
  padding: 8px 20px !important;
  transition: opacity .15s ease;
}
.stButton > button:hover { opacity: .87; }
.stButton > button:focus-visible { outline: 2px solid var(--terra); outline-offset: 2px; }

/* Download button */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  color: var(--terra) !important;
  border: 1px solid var(--terra) !important;
  border-radius: var(--radius) !important;
  font-weight: 500;
  transition: background .15s ease, color .15s ease;
}
[data-testid="stDownloadButton"] > button:hover {
  background: var(--terra) !important;
  color: #fff !important;
}
[data-testid="stDownloadButton"] > button:focus-visible {
  outline: 2px solid var(--terra); outline-offset: 2px;
}

/* File uploader */
[data-testid="stFileUploader"] {
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface) !important;
  padding: 12px !important;
}
[data-testid="stFileUploader"]:focus-within {
  border-color: var(--terra) !important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--terra) !important;
  border-color: var(--terra) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:focus-visible {
  outline: 2px solid var(--terra); outline-offset: 2px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* Divider */
hr { border-color: var(--border) !important; opacity: 1 !important; }

/* Alert / info boxes */
[data-testid="stAlert"] {
  border-radius: var(--radius) !important;
  border-left-width: 4px !important;
}

/* Headings */
h1, h2, h3 { color: var(--text) !important; font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 1.6rem !important; font-weight: 700; letter-spacing: -0.03em; }
h2 { font-size: 1.15rem !important; }
h3 { font-size: 0.95rem !important; color: var(--muted) !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }

/* Caption / small text */
.stMarkdown p, .stCaption { color: var(--muted) !important; }

/* Code blocks */
code { background: var(--surface) !important; border-radius: 3px; font-size: 0.85em; padding: 1px 5px; }
pre { background: var(--surface) !important; border: 1px solid var(--border); border-radius: var(--radius); }

/* Spinner */
[data-testid="stSpinner"] { color: var(--terra) !important; }

/* ── Reduced-motion ── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    transition-duration: .01ms !important;
  }
}
</style>
"""


# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading image model…")
def load_image_model():
    try:
        model, meta = load_model(config.MODELS_DIR, "emocraft_image")
        return model, meta, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Failed to load image model: {e}"


@st.cache_resource(show_spinner="Loading video model…")
def load_video_model():
    try:
        model, meta = load_model(config.MODELS_DIR, "emocraft_video")
        return model, meta, None
    except FileNotFoundError:
        return None, None, None   # Not trained yet — silently skip
    except Exception as e:
        return None, None, f"Failed to load video model: {e}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def bgr_to_rgb_bytes(bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", rgb)
    return buf.tobytes() if ok else b""


def display_confidence_bars(all_scores: dict[str, float]) -> None:
    """Render a clean confidence bar for each emotion using the design palette."""
    sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top_label = sorted_emotions[0][0] if sorted_emotions else ""

    for emotion, score in sorted_emotions:
        pct = score * 100
        is_top = emotion == top_label
        bar_color = "#AD7059" if is_top else "#C4A898"   # terra for winner, muted for rest
        label_weight = "600" if is_top else "400"

        st.markdown(
            f"""
            <div style="margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.83rem;margin-bottom:3px;color:#272923">
                <span style="font-weight:{label_weight}">{emotion}</span>
                <span style="color:#6B6960">{pct:.1f}%</span>
              </div>
              <div style="background:#E3E0D7;border-radius:3px;height:7px;overflow:hidden">
                <div style="width:{pct:.1f}%;background:{bar_color};height:100%;
                            border-radius:3px;transition:width .3s ease"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def result_card(label: str, confidence: float) -> None:
    """A clean result card: left terracotta border, label and confidence."""
    st.markdown(
        f"""
        <div style="
          border-left: 4px solid #AD7059;
          background: #EDEAE3;
          border-radius: 0 6px 6px 0;
          padding: 18px 20px;
          margin: 4px 0 12px 0;
        ">
          <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;
                      text-transform:uppercase;color:#6B6960;margin-bottom:4px">
            Detected expression
          </div>
          <div style="font-size:1.7rem;font-weight:700;color:#272923;
                      letter-spacing:-0.02em;line-height:1.1">
            {label.capitalize()}
          </div>
          <div style="font-size:0.88rem;color:#6B6960;margin-top:4px">
            {confidence:.0%} confidence
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='padding:4px 0 2px'>"
            "<span style='font-size:1.25rem;font-weight:700;letter-spacing:-0.02em;color:#272923'>EmoCraft</span>"
            "<br><span style='font-size:0.78rem;color:#6B6960'>Facial expression recognition</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Model status ──
        st.markdown("### Models")
        img_model, img_meta, img_err = load_image_model()
        vid_model, vid_meta, vid_err = load_video_model()

        if img_model is not None:
            st.success(
                f"Image model ready  \n"
                f"Classes: {', '.join(img_meta['class_names'])}"
            )
        elif img_err:
            st.error(f"Image model — {img_err}")
            st.caption("Run `python train_image.py` to train the image model.")
        else:
            st.warning("Image model not trained yet.")

        if vid_model is not None:
            st.success(
                f"Video model ready  \n"
                f"Classes: {', '.join(vid_meta['class_names'])}"
            )
        elif vid_err:
            st.warning(f"Video model — {vid_err}")
        else:
            st.info("Video model not trained.  \nRun `python train_video.py`.")

        st.divider()

        # ── Face detector ──
        st.markdown("### Face detector")
        face_path = config.FACE_DETECTOR_MODEL
        if face_path.exists():
            st.success(f"YuNet ONNX  \n{face_path.stat().st_size:,} bytes")
        else:
            st.error(f"YuNet model not found:  \n`{face_path}`")

        st.divider()

        # ── Disclaimer ──
        st.caption(
            "**Note on predictions** — Results describe visible expression categories "
            "in the image or video. They do not establish a person's true feelings "
            "and are not a clinical assessment."
        )
        st.caption(
            "**Datasets** — FER2013 (research use) · CREMA-D (ODbL) · RAVDESS (CC BY-NC-SA 4.0)  \n"
            "Accuracy reflects held-out test data; real-world performance may differ."
        )


# ── Image tab ─────────────────────────────────────────────────────────────────

def render_image_tab() -> None:
    st.header("Image analysis")
    st.markdown(
        "<p style='color:#6B6960;margin-top:-8px;margin-bottom:20px'>"
        "Upload a clear photo containing one visible face. "
        "Best results with front-facing, well-lit images."
        "</p>",
        unsafe_allow_html=True,
    )

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
            f"Face detector model not found at `{face_path}`.  \n"
            "Run `python smoke_test.py` to download it automatically."
        )
        return

    uploaded = st.file_uploader(
        "Drop or select an image — JPEG, PNG, BMP, WebP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if uploaded is None:
        st.info("Upload an image above to see the emotion analysis.")
        return

    image_bytes = uploaded.read()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Input image")
        st.image(image_bytes, use_container_width=True)

    with st.spinner("Detecting face and predicting emotion…"):
        result = predict_image(image_bytes, img_model, img_meta, face_path)

    with col2:
        st.markdown("#### Result")

        if result["error"]:
            st.error(result["error"])
            if result.get("annotated_bgr") is not None:
                st.image(bgr_to_rgb_bytes(result["annotated_bgr"]), use_container_width=True)
            return

        result_card(result["label"], result["confidence"])

        if not result["face_found"]:
            st.caption(
                "No bounding box detected — the entire image was treated as a pre-cropped face."
            )

        st.markdown("#### Confidence breakdown")
        display_confidence_bars(result["all_scores"])

    if result["annotated_bgr"] is not None:
        st.divider()
        st.markdown("#### Annotated image")
        st.image(bgr_to_rgb_bytes(result["annotated_bgr"]), use_container_width=True)


# ── Video tab ─────────────────────────────────────────────────────────────────

def render_video_tab() -> None:
    st.header("Video analysis")
    st.markdown(
        "<p style='color:#6B6960;margin-top:-8px;margin-bottom:20px'>"
        "Upload a short video clip. EmoCraft samples frames, detects faces in each, "
        "and returns an overall expression and a per-frame breakdown."
        "</p>",
        unsafe_allow_html=True,
    )

    vid_model, vid_meta, vid_err = load_video_model()
    img_model, img_meta, img_err = load_image_model()

    if vid_model is not None:
        active_model, active_meta, model_label = vid_model, vid_meta, "video model"
    elif img_model is not None:
        active_model, active_meta, model_label = img_model, img_meta, "image model (fallback)"
        st.warning(
            "No video-specific model found — using the image model as a fallback.  \n"
            "Train a dedicated model with `python train_video.py` for better results."
        )
    else:
        st.error("No model available. Train at least the image model: `python train_image.py`")
        return

    face_path = config.FACE_DETECTOR_MODEL
    if not face_path.exists():
        st.error(f"Face detector model not found at `{face_path}`.")
        return

    st.caption(
        f"Active: **{model_label}** — "
        f"classes: {', '.join(active_meta['class_names'])}"
    )
    st.divider()

    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        target_fps = st.slider(
            "Frames sampled per second",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            help="Higher values process more frames (slower but more stable).",
        )
    with col_b:
        max_frames = st.slider(
            "Maximum frames to process",
            min_value=10, max_value=120, value=60,
            help="Cap on the total number of sampled frames.",
        )

    uploaded = st.file_uploader(
        "Drop or select a video — MP4, AVI, MOV, MKV",
        type=["mp4", "avi", "mov", "mkv", "flv"],
        key="video_upload",
    )

    if uploaded is None:
        st.info("Upload a video above to begin.")
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded.read())
        tmp_path = Path(tf.name)

    try:
        with st.spinner("Processing video — sampling frames and detecting emotions…"):
            t0 = time.time()
            result = predict_video(
                tmp_path, active_model, active_meta, face_path,
                target_fps=target_fps, max_frames=max_frames,
            )
            elapsed = time.time() - t0

        if result["error"]:
            st.error(result["error"])
            return

        # ── Summary banner ──
        st.success(
            f"Processed {result['frames_processed']} frames "
            f"({result['frames_with_face']} with a detected face) in {elapsed:.1f} s"
        )

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("#### Overall prediction")
            result_card(result["label"], result["confidence"])
            st.markdown("#### Mean confidence per expression")
            display_confidence_bars(result["all_scores"])

        with col2:
            st.markdown("#### Per-frame results")
            if result["per_frame"]:
                frame_data = [
                    {
                        "Frame": r["frame_idx"],
                        "Expression": r["label"].capitalize(),
                        "Confidence": f"{r['confidence']:.0%}",
                    }
                    for r in result["per_frame"]
                ]
                st.dataframe(frame_data, use_container_width=True, height=300)
            else:
                st.info("No frames with detected faces.")

        # ── Annotated video ──
        st.divider()
        st.markdown("#### Annotated video")
        if result["annotated_frames_bgr"]:
            out_path = tmp_path.parent / f"annotated_{tmp_path.stem}.mp4"
            written = write_annotated_video(result["annotated_frames_bgr"], out_path, fps=target_fps)
            if written and written.exists():
                with open(written, "rb") as f:
                    st.download_button(
                        "Download annotated video (MP4)",
                        data=f.read(),
                        file_name=f"emocraft_annotated_{uploaded.name}",
                        mime="video/mp4",
                    )
                # Sample frames preview
                st.markdown("#### Sample frames")
                sample_frames = result["annotated_frames_bgr"][
                    :: max(1, len(result["annotated_frames_bgr"]) // 6)
                ][:6]
                cols = st.columns(min(3, len(sample_frames)))
                for i, frame in enumerate(sample_frames[:3]):
                    with cols[i % 3]:
                        st.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
            else:
                st.warning("Could not write the annotated video file.")
        else:
            st.info("No annotated frames to export — no faces detected in sampled frames.")

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── About tab ─────────────────────────────────────────────────────────────────

def render_about_tab() -> None:
    st.header("About EmoCraft")

    st.info(
        "**On predictions** — Results describe *visible expression categories* "
        "detected in an image or video. They do not establish a person's true feelings "
        "and should not be used as a clinical, psychological, or legal assessment."
    )

    st.markdown("""
EmoCraft is a facial emotion recognition system built with a convolutional neural
network trained on the FER2013 benchmark dataset. It uses OpenCV's YuNet ONNX face
detector for robust face localisation without Haar cascades.

---

#### Architecture

- Custom lightweight CNN — 5 convolutional layers + global average pooling
- Input: 48 × 48 grayscale face crops
- Face detection: OpenCV YuNet (`FaceDetectorYN`)
- Training: TensorFlow / Keras with class-weighted loss, early stopping, learning-rate scheduling

---

#### Datasets & licences

| Dataset | Purpose | Classes | Licence |
|---------|---------|---------|---------|
| FER2013 | Image model training | 7 (angry, disgust, fear, happy, neutral, sad, surprise) | Research use (Kaggle competition) |
| CREMA-D | Video model training (preferred) | 6 (no surprise) | Open Database Licence (ODbL) v1.0 |
| RAVDESS | Pilot / smoke-test fallback (2 actors only) | 7 | CC BY-NC-SA 4.0 |

**FER2013** — Goodfellow, I. J., et al. (2013). *Challenges in Representation Learning:
A report on three machine learning contests.* ICONIP 2013.

**CREMA-D** — Cao, H., Cooper, D. G., Kuchera-Morin, J., et al. (2014).
*CREMA-D: Crowd-sourced emotional multimodal actors dataset.*
IEEE Transactions on Affective Computing.

**RAVDESS** — Livingstone, S. R. & Russo, F. A. (2018).
*The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).*
PLoS ONE 13(5): e0196391.

---

#### Label alignment

- FER2013 has 7 classes; CREMA-D has 6 (no *surprise*).
- Each model's class list is saved alongside its weights and loaded at inference time.
- The app always shows which model is active; no silent remapping occurs.

---

#### Honest limitations

- FER2013 CNN baselines typically reach **60–68 % test accuracy** — this benchmark is
  considered hard; higher published numbers often use extra data or ensembles.
- The RAVDESS 2-actor pilot is a smoke test only and does **not** generalise to held-out actors.
- Real-world accuracy on unconstrained images or video can differ substantially from
  benchmark results.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Inject global styles
    st.markdown(_CSS, unsafe_allow_html=True)

    render_sidebar()

    # ── Page header ──
    st.markdown(
        "<h1 style='margin-bottom:2px'>EmoCraft</h1>"
        "<p style='color:#6B6960;margin-top:0;margin-bottom:24px;font-size:0.95rem'>"
        "Analyse visible facial expressions in images and video."
        "</p>",
        unsafe_allow_html=True,
    )

    tab_img, tab_vid, tab_about = st.tabs(["Image", "Video", "About"])

    with tab_img:
        render_image_tab()

    with tab_vid:
        render_video_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
