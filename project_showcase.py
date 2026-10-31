"""Single-file Streamlit showcase for EmoCraft.

Run from the project root with:
    streamlit run project_showcase.py

The dashboard reads the evaluation reports and plots already saved in outputs/.
It does not train models or change project files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
RAVDESS_OUTPUTS = ROOT / "RAVDESS" / "run_output"

st.set_page_config(
    page_title="EmoCraft | Project Showcase",
    page_icon="🎭",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1280px; padding-top: 1.8rem; padding-bottom: 3rem;}
      div[data-testid="stMetric"] {background: #f4f7fb; border: 1px solid #e5eaf1;
        border-radius: 12px; padding: 16px 18px;}
      .subtle {color: #657184; font-size: 0.94rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict[str, Any]:
    """Load one project report, returning an empty mapping if it is unavailable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def percent(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "Not recorded"


def metric_row(
    name: str,
    task: str,
    dataset: str,
    accuracy: Any,
    macro_f1: Any = None,
    balanced_accuracy: Any = None,
    status: str = "",
    note: str = "",
) -> dict[str, Any]:
    return {
        "Model": name,
        "Task": task,
        "Evaluation data": dataset,
        "Accuracy (%)": round(float(accuracy) * 100, 2) if accuracy is not None else None,
        "Macro-F1 (%)": round(float(macro_f1) * 100, 2) if macro_f1 is not None else None,
        "Balanced accuracy (%)": (
            round(float(balanced_accuracy) * 100, 2)
            if balanced_accuracy is not None
            else None
        ),
        "App status": status,
        "Notes": note,
    }


def class_metrics_table(report: dict[str, Any], class_names: list[str]) -> pd.DataFrame:
    rows = []
    for label in class_names:
        scores = report.get(label)
        if not isinstance(scores, dict):
            continue
        rows.append(
            {
                "Expression": label.title(),
                "Precision (%)": round(float(scores.get("precision", 0)) * 100, 1),
                "Recall (%)": round(float(scores.get("recall", 0)) * 100, 1),
                "F1 (%)": round(float(scores.get("f1-score", 0)) * 100, 1),
                "Examples": int(scores.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def image_gallery(
    paths: list[Path],
    columns: int = 2,
    captions: dict[str, str] | None = None,
) -> None:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        st.info("No saved image outputs were found in this checkout.")
        return
    for start in range(0, len(existing), columns):
        row = st.columns(columns)
        for col, path in zip(row, existing[start : start + columns]):
            with col:
                relative = str(path.relative_to(ROOT))
                caption = (captions or {}).get(relative, path.name)
                st.image(str(path), caption=caption, use_container_width=True)


image_report = read_json(OUTPUTS / "image_eval_summary.json")
video_report = read_json(OUTPUTS / "video_eval_summary.json")
legacy_video_report = read_json(RAVDESS_OUTPUTS / "run_summary.json")

# Recorded actor-disjoint Keras pilot metrics from the train_video.py evaluation.
# This is shown as a historical comparison; the app uses HSEmotion for video.
keras_video_pilot = {
    "accuracy": 0.3269230769230769,
    "balanced_accuracy": 0.30357142857142855,
    "macro_f1": 0.2119140989729225,
    "dataset": "RAVDESS actor 06 held out (52 clips)",
}

class_names = image_report.get(
    "class_names",
    ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
)
image_per_class = image_report.get("per_class", {})
video_per_class = video_report.get("per_class", {})

comparison_rows = [
    metric_row(
        "FER2013 image CNN (raw)",
        "Image",
        "FER2013 official test",
        image_report.get("test_accuracy_keras_uncalibrated"),
        status="Baseline",
        note="Flip-averaged CNN before class calibration or HSEmotion blend.",
    ),
    metric_row(
        "FER2013 image CNN (calibrated)",
        "Image",
        "FER2013 official test",
        image_report.get("test_accuracy_calibrated_keras_only"),
        status="Comparison",
        note="CNN-only scores with the final class offsets.",
    ),
    metric_row(
        "CNN + HSEmotion image ensemble",
        "Image",
        "FER2013 official test",
        image_report.get("test_accuracy"),
        image_report.get("test_macro_f1"),
        image_report.get("test_balanced_accuracy"),
        status="Active",
        note="Validation-selected ensemble and angry/fear/sad calibration.",
    ),
    metric_row(
        "Original RAVDESS PyTorch demo",
        "Video",
        legacy_video_report.get("dataset", "RAVDESS pilot"),
        legacy_video_report.get("clip_accuracy"),
        legacy_video_report.get("clip_macro_f1"),
        legacy_video_report.get("clip_balanced_accuracy"),
        status="Historical",
        note="Two-actor demonstration model; included to show the project progression.",
    ),
    metric_row(
        "FER CNN transfer video pilot",
        "Video",
        keras_video_pilot["dataset"],
        keras_video_pilot["accuracy"],
        keras_video_pilot["macro_f1"],
        keras_video_pilot["balanced_accuracy"],
        status="Pilot; not active",
        note="Recorded train_video.py run; two actors only. The app skips this model because its clip accuracy was weak.",
    ),
    metric_row(
        "HSEmotion frame model",
        "Video",
        video_report.get("dataset", "RAVDESS actor 06 pilot"),
        video_report.get("accuracy"),
        video_report.get("macro_f1"),
        video_report.get("balanced_accuracy"),
        status="Active",
        note="Mean class probabilities over sampled video frames.",
    ),
]
comparison_df = pd.DataFrame(comparison_rows)

st.title("🎭 EmoCraft — Project Showcase")
st.caption("Facial-expression recognition from still images and sampled video frames")
st.markdown(
    """
    This presentation summarizes the image and video models, their evaluation results,
    class-level performance, and saved training outputs. The dashboard reads existing
    project reports; it does not start model training.
    """
)

overview_tab, comparison_tab, image_tab, video_tab, artifacts_tab = st.tabs(
    ["Overview", "Model comparison", "Image model", "Video model", "Saved outputs"]
)

with overview_tab:
    active_image_accuracy = image_report.get("test_accuracy")
    active_video_accuracy = video_report.get("accuracy")
    image_count = int(image_report.get("per_class", {}).get("accuracy", {}).get("support", 0))
    video_count = int(video_report.get("clips_evaluated", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active image model", percent(active_image_accuracy))
    c2.metric("Active video model", percent(active_video_accuracy))
    c3.metric("Image test examples", f"{image_count:,}")
    c4.metric("Video pilot clips", f"{video_count:,}")

    st.subheader("What the application does")
    flow = st.columns(3)
    flow[0].markdown("**1 · Detect a face**\n\nOpenCV YuNet finds the largest face in the image or sampled frame.")
    flow[1].markdown("**2 · Score expressions**\n\nThe image path blends the FER2013 CNN with HSEmotion. The video path scores frames with HSEmotion.")
    flow[2].markdown("**3 · Summarize results**\n\nClass scores are shown for images and averaged across sampled frames for a clip-level result.")

    st.info(
        "The available video evaluation uses only RAVDESS actors 05 and 06. "
        "Actor 06 was consulted during model comparisons, so its score is a project "
        "diagnostic rather than a reliable estimate for new people."
    )

    st.subheader("Run the showcase")
    st.code("streamlit run project_showcase.py", language="powershell")
    st.caption("Install the project dependencies first with: python -m pip install -r requirements.txt")

with comparison_tab:
    st.subheader("Accuracy by model")
    st.caption(
        "Rows use different datasets and splits. Compare models within the same task and dataset; "
        "image and video percentages are not directly comparable."
    )
    visible_df = comparison_df.dropna(subset=["Accuracy (%)"]).copy()
    if visible_df.empty:
        st.warning("Model evaluation reports are missing. Check the outputs/ directory.")
    else:
        chart_df = visible_df.set_index("Model")[["Accuracy (%)"]]
        st.bar_chart(chart_df, height=340, y_label="Accuracy (%)", x_label="Model")
        st.dataframe(visible_df, use_container_width=True, hide_index=True)

    st.subheader("Reading the results")
    st.markdown(
        f"""
        - Image ensemble accuracy: **{percent(image_report.get('test_accuracy'))}** on the
          official FER2013 test set. Its reported test metrics were consulted during model
          comparisons, so treat them as diagnostics.
        - Video HSEmotion accuracy: **{percent(video_report.get('accuracy'))}** on
          {video_count} held-out RAVDESS actor-06 clips. Only two actors are available, so
          this is not a production generalization estimate.
        - The image model does **not** reach the earlier 85% target on FER2013.
        """
    )

with image_tab:
    st.subheader("Image model — FER2013")
    st.write(image_report.get("evaluation_model", "FER2013 image CNN"))
    image_metrics = st.columns(4)
    image_metrics[0].metric("Accuracy", percent(image_report.get("test_accuracy")))
    image_metrics[1].metric("Macro-F1", percent(image_report.get("test_macro_f1")))
    image_metrics[2].metric("Balanced accuracy", percent(image_report.get("test_balanced_accuracy")))
    image_metrics[3].metric("HSEmotion blend", percent(image_report.get("hse_ensemble_weight")))

    st.markdown("#### Per-expression test results")
    image_table = class_metrics_table(image_per_class, class_names)
    if not image_table.empty:
        st.dataframe(image_table, use_container_width=True, hide_index=True)
        st.bar_chart(image_table.set_index("Expression")[["Recall (%)", "F1 (%)"]], height=300)
    else:
        st.warning("Per-class image metrics were not found.")

    if image_report.get("test_metric_status"):
        st.warning(image_report["test_metric_status"])

    st.markdown("#### Saved image outputs")
    image_gallery(
        [
            OUTPUTS / "image_confusion_matrix.png",
            OUTPUTS / "image_training_curves.png",
        ],
        columns=2,
        captions={
            "outputs/video_confusion_matrix.png": "Active HSEmotion video pilot — RAVDESS actor 06",
            "outputs/video_training_curves.png": "Legacy Keras transfer pilot training curves (not the active HSEmotion model)",
            "RAVDESS/run_output/confusion_matrix.png": "Original PyTorch RAVDESS demo confusion matrix",
            "RAVDESS/run_output/training_curves.png": "Original PyTorch RAVDESS demo training curves",
            "RAVDESS/run_output/sample_prediction.png": "Original PyTorch demo sample prediction",
        },
    )

with video_tab:
    st.subheader("Video model — HSEmotion frame scores")
    st.write(video_report.get("model", "HSEmotion ONNX model"))
    video_metrics = st.columns(4)
    video_metrics[0].metric("Clip accuracy", percent(video_report.get("accuracy")))
    video_metrics[1].metric("Macro-F1", percent(video_report.get("macro_f1")))
    video_metrics[2].metric("Balanced accuracy", percent(video_report.get("balanced_accuracy")))
    video_metrics[3].metric("Clips evaluated", str(video_count))

    st.markdown("#### Per-expression video results")
    video_table = class_metrics_table(video_per_class, video_report.get("class_names", class_names))
    if not video_table.empty:
        st.dataframe(video_table, use_container_width=True, hide_index=True)
        st.bar_chart(video_table.set_index("Expression")[["Recall (%)", "F1 (%)"]], height=300)
    else:
        st.warning("Per-class video metrics were not found.")

    if video_report.get("warning"):
        st.warning(video_report["warning"])

    st.markdown("#### Saved video outputs")
    image_gallery(
        [
            OUTPUTS / "video_confusion_matrix.png",
            OUTPUTS / "video_training_curves.png",
            RAVDESS_OUTPUTS / "confusion_matrix.png",
            RAVDESS_OUTPUTS / "training_curves.png",
            RAVDESS_OUTPUTS / "sample_prediction.png",
        ],
        columns=2,
    )

with artifacts_tab:
    st.subheader("Evaluation reports")
    with st.expander("Image evaluation JSON", expanded=False):
        st.json(image_report)
    with st.expander("Video evaluation JSON", expanded=False):
        st.json(video_report)
    with st.expander("Original RAVDESS demo JSON", expanded=False):
        if legacy_video_report:
            st.json(legacy_video_report)
        else:
            st.info("No original RAVDESS demo summary was found.")

    st.subheader("Project artifacts")
    artifact_roots = [OUTPUTS, MODELS, RAVDESS_OUTPUTS]
    artifact_rows = []
    for root in artifact_roots:
        if root.is_dir():
            for path in sorted(root.iterdir()):
                if path.is_file() and path.suffix.lower() in {".png", ".json", ".keras", ".onnx", ".pth", ".h5"}:
                    artifact_rows.append(
                        {
                            "Location": str(path.relative_to(ROOT)),
                            "Type": path.suffix.lower().lstrip("."),
                            "Size (MB)": round(path.stat().st_size / (1024 * 1024), 2),
                        }
                    )
    if artifact_rows:
        st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No saved artifacts were found.")

    st.caption(
        "FER2013 image scores and RAVDESS video scores come from separate datasets. "
        "The RAVDESS pilot contains two actors and should not be presented as a generalization benchmark."
    )
