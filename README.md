# EmoCraft — Facial Emotion Recognition

A complete local application for facial emotion recognition from images and videos.
Built on TensorFlow/Keras with a custom CNN, OpenCV YuNet face detection, and a Streamlit UI.

---

## Quick start (One-Click Launch)

Simply double-click **`run.bat`** in the project folder, or run:

```bat
cd C:\Users\aweso\OneDrive\Desktop\sun
.\run.bat
```

or via Python:

```bat
python run.py
```

This automatically:
1. Verifies the environment and models
2. Automatically opens your default web browser to **http://localhost:8501**
3. Starts the Streamlit application server

---

## Setup

### Prerequisites

- Python 3.14 (installed at `C:\Users\aweso\AppData\Local\Programs\Python\Python314\`)
- Windows 10 / 11

### Install dependencies

```bat
cd C:\Users\aweso\OneDrive\Desktop\sun
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe -m pip install -r requirements.txt
```

### Run the smoke test (checks environment + dataset)

```bat
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe smoke_test.py
```

---

## Datasets

### 1. FER2013 (image model — required)

- **Classes**: angry, disgust, fear, happy, neutral, sad, surprise (7 classes)
- **Size**: ~60 MB (~35,000 images, 48×48 grayscale)
- **Licence**: Research use (Kaggle competition dataset)
- **Split used**: official `train/` (28,709 images) and `test/` (7,178 images)
  - 10% of `train/` is withheld for validation (stratified by class)
  - `test/` is **never** used during training

**How to get it:**

```bat
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe -m kaggle datasets download -d msambare/fer2013 -p datasets\fer2013
```

Then unzip `datasets\fer2013\fer2013.zip` in place. The result should be:
```
datasets\fer2013\train\angry\  ...
datasets\fer2013\test\angry\   ...
```

Set `EMOCRAFT_FER2013_DIR` if you place it elsewhere.

**Citation:**
> Goodfellow, I. J., et al. (2013). Challenges in Representation Learning.
> NIPS 2013 Deep Learning Workshop.

---

### 2. CREMA-D (video model — strongly preferred)

- **Classes**: angry, disgust, fear, happy, neutral, sad (6 classes — **no surprise**)
- **Actors**: 91 actors (actor-disjoint train/val/test split)
- **Size**: ~8.5 GB (requires Git LFS)
- **Licence**: Open Database Licence (ODbL) v1.0

**How to get it:**

```bat
git clone https://github.com/CheyneyComputerScience/CREMA-D
cd CREMA-D
git lfs pull
```

Then set the path:
```bat
set EMOCRAFT_CREMAD_DIR=C:\path\to\CREMA-D
```

Or edit `emocraft/config.py` and update `CREMAD_DIR`.

**Citation:**
> Cao, H., Cooper, D. G., Kuchera-Morin, J., et al. (2014).
> CREMA-D: Crowd-sourced emotional multimodal actors dataset.
> IEEE Transactions on Affective Computing, 5(4), 377–390.

---

### 3. RAVDESS (pilot / smoke-test fallback — 2 actors only)

> ⚠  **Warning**: Only actors 05 and 06 are currently downloaded.
> Two actors cannot provide reliable generalization estimates.
> Results from this dataset are labeled as demo/pilot only.

- **Location**: `RAVDESS\Video_Speech_Actor_05.zip`, `Video_Speech_Actor_06.zip`
- **Licence**: CC BY-NC-SA 4.0

**Citation:**
> Livingstone, S. R. & Russo, F. A. (2018). The Ryerson Audio-Visual Database of
> Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.

---

## Training

### Image model (FER2013)

```bat
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe train_image.py
```

Options:
```
--epochs N          (default: 50)
--batch-size N      (default: 64)
--fer-dir PATH      (default: datasets\fer2013)
--models-dir PATH   (default: models\)
--no-verify         Skip per-image file validation
```

Saved artefacts:
- `models\emocraft_image.keras`      — model weights
- `models\emocraft_image_meta.json`  — class list + image size
- `outputs\image_training_curves.png`
- `outputs\image_confusion_matrix.png`
- `outputs\image_eval_summary.json`

### Video model (CREMA-D or RAVDESS fallback)

```bat
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe train_video.py
```

Options:
```
--cremad-dir PATH   (default: datasets\cremad)
--ravdess-dir PATH  (default: RAVDESS\)
--epochs N          (default: 50)
```

Saved artefacts:
- `models\emocraft_video.keras`
- `models\emocraft_video_meta.json`
- `outputs\video_training_curves.png`
- `outputs\video_confusion_matrix.png`
- `outputs\video_eval_summary.json`

---

## Launch the application

```bat
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe -m streamlit run app.py
```

Open: **http://localhost:8501**

The app has three tabs:
- **📷 Image**: upload a face photo → get emotion label, confidence, annotated image
- **🎬 Video**: upload a video → per-frame emotion analysis, aggregate result, downloadable annotated clip
- **ℹ About**: dataset citations, architecture notes, limitations

---

## Project structure

```
sun\
├── app.py                      Streamlit application
├── train_image.py              FER2013 image model training script
├── train_video.py              CREMA-D / RAVDESS video model training script
├── smoke_test.py               End-to-end environment + dataset smoke test
├── requirements.txt            Python dependencies
├── README.md                   This file
│
├── emocraft\                   Core library
│   ├── __init__.py
│   ├── config.py               Configurable paths and hyperparameters
│   ├── model.py                CNN architecture + save/load helpers
│   ├── face_detector.py        YuNet face detection (cv2.FaceDetectorYN)
│   ├── inference.py            Image + video prediction pipeline
│   ├── fer2013_data.py         FER2013 dataset utilities
│   ├── cremad_data.py          CREMA-D dataset utilities
│   └── ravdess_data.py         RAVDESS dataset utilities (pilot fallback)
│
├── datasets\
│   ├── fer2013\                FER2013 dataset (train/ + test/)
│   └── cremad\                 CREMA-D dataset (AudioVisual/)
│
├── models\                     Saved model weights and metadata
│   ├── face_detection_yunet_2023mar.onnx
│   ├── emocraft_image.keras
│   ├── emocraft_image_meta.json
│   ├── emocraft_video.keras    (if video model trained)
│   └── emocraft_video_meta.json
│
├── outputs\                    Training curves, confusion matrices, summaries
│
├── RAVDESS\                    RAVDESS ZIPs (actors 05 and 06)
│   ├── Video_Speech_Actor_05.zip
│   ├── Video_Speech_Actor_06.zip
│   └── run_output\             Previous pilot run artefacts
│
├── EmoCraft_RAVDESS_Run.py     Original RAVDESS pilot script (unmodified)
└── EmoCraft_Image_Video_RAVDESS.ipynb   Original notebook (unmodified)
```

---

## Configurable environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMOCRAFT_FER2013_DIR` | `datasets/fer2013` | FER2013 root (must contain `train/` and `test/`) |
| `EMOCRAFT_CREMAD_DIR` | `datasets/cremad` | CREMA-D repo clone root |
| `EMOCRAFT_RAVDESS_DIR` | `RAVDESS/` | RAVDESS ZIPs directory |
| `EMOCRAFT_MODELS_DIR` | `models/` | Where trained models are saved/loaded |
| `EMOCRAFT_FACE_MODEL` | `models/face_detection_yunet_2023mar.onnx` | YuNet ONNX path |
| `EMOCRAFT_IMAGE_SIZE` | `48` | CNN input size (px) |
| `EMOCRAFT_BATCH_SIZE` | `64` | Training batch size |
| `EMOCRAFT_EPOCHS` | `50` | Max training epochs |
| `EMOCRAFT_LR` | `0.001` | Initial learning rate |
| `EMOCRAFT_SEED` | `42` | Random seed for reproducibility |

---

## Label alignment

| Label | FER2013 | CREMA-D | RAVDESS |
|-------|---------|---------|---------|
| angry | ✅ | ✅ | ✅ |
| disgust | ✅ | ✅ | ✅ |
| fear | ✅ | ✅ | ✅ |
| happy | ✅ | ✅ | ✅ |
| neutral | ✅ | ✅ | ✅ |
| sad | ✅ | ✅ | ✅ |
| surprise | ✅ | ❌ (absent) | ✅ |
| calm | ❌ | ❌ | ✅ (excluded) |

Models trained on CREMA-D will not output `surprise`; models trained on FER2013 will.
Each model's class list is saved in its `_meta.json` file and loaded at inference time.

---

## Honest performance expectations

FER2013 is a notoriously hard benchmark:
- Human agreement is ~65%
- CNN baselines typically achieve **60–68% accuracy** on the test set
- Class imbalance (disgust has only ~1.5% of samples) makes balanced accuracy lower

The **RAVDESS 2-actor pilot** achieved:
- Clip accuracy: **7.7%** (near-random)
- This is expected — two actors cannot train a generalizing model

For the **CREMA-D** video model (if trained):
- Actor-disjoint evaluation is the right benchmark — new actors will be harder
- Typical published baselines: 55–70% clip-level accuracy depending on modality

---

## Troubleshooting

**`No module named 'tensorflow'`**
```bat
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe -m pip install tensorflow
```

**`No module named 'streamlit'`**
```bat
C:\Users\aweso\AppData\Local\Programs\Python\Python314\python.exe -m pip install streamlit
```

**FER2013 `403 Forbidden` from kaggle competitions download**
Use the dataset download instead:
```bat
python -m kaggle datasets download -d msambare/fer2013 -p datasets\fer2013
```

**Face detector not finding faces**
- Try images with clearly visible, front-facing faces
- Very small images (< 50px) may not be detected
- The YuNet detector works on images ≥ 8×8 px but performs best on ≥ 100px faces

