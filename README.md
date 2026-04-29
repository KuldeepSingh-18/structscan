# 🏗️ StructScan — Early Structural Damage Detection System

A real-time structural damage detection application using Computer Vision and Deep Learning.
Supports **live webcam**, **image upload**, and **video upload** with Grad-CAM visualization,
severity scoring, and civil engineering repair recommendations.

---

## 📁 Project Structure

```
structural-damage-detector/
│
├── backend/                    ← FastAPI server + ML inference
│   ├── __init__.py
│   ├── main.py                 ← API routes + WebSocket endpoint
│   ├── model.py                ← MobileNetV2 inference + sliding window
│   ├── gradcam.py              ← Grad-CAM heatmap generation
│   └── alert.py                ← Severity levels + repair suggestions
│
├── frontend/                   ← Web UI (served by FastAPI)
│   ├── index.html              ← Main page (industrial monitor aesthetic)
│   ├── style.css               ← All styling
│   └── app.js                  ← WebSocket, camera, upload logic
│
├── train/                      ← Offline training scripts
│   ├── prepare_dataset.py      ← Organize raw images into train/val/test
│   ├── train_model.py          ← Full training pipeline (MobileNetV2)
│   └── evaluate_model.py       ← Metrics, confusion matrix, ROC curve
│
├── models/                     ← Saved model files (auto-created)
│   └── crack_model.h5          ← Trained model (created after training)
│
├── dataset/                    ← Dataset folder (you fill this)
│   ├── train/
│   │   ├── cracked/            ← Put cracked images here
│   │   └── non_cracked/        ← Put intact surface images here
│   ├── val/
│   │   ├── cracked/
│   │   └── non_cracked/
│   └── test/
│       ├── cracked/
│       └── non_cracked/
│
├── run.py                      ← Start the app ← MAIN ENTRY POINT
├── requirements.txt
└── README.md
```

---

## ⚙️ Windows Setup (Step by Step)

### STEP 1 — Install Python 3.10

1. Go to: https://www.python.org/downloads/release/python-3100/
2. Download **Windows installer (64-bit)**
3. Run installer — ✅ CHECK **"Add Python to PATH"** before clicking Install
4. Verify in Command Prompt:
   ```
   python --version
   ```
   Should show: `Python 3.10.x`

> ⚠️ **Use Python 3.10 specifically** — TensorFlow 2.13 requires it on Windows.
> Python 3.11+ will NOT work with TensorFlow 2.13.

---

### STEP 2 — Install VSCode

1. Download from: https://code.visualstudio.com/
2. Install with default settings
3. Open VSCode → Install these extensions:
   - **Python** (by Microsoft)
   - **Pylance** (by Microsoft)

---

### STEP 3 — Open Project in VSCode

1. Copy the `structural-damage-detector/` folder to your desired location (e.g., `C:\Projects\`)
2. Open VSCode → **File → Open Folder** → Select `structural-damage-detector`

---

### STEP 4 — Create Virtual Environment

Open VSCode Terminal (`Ctrl + `` ` `` `):

```bash
# Create virtual environment
python -m venv venv

# Activate it (IMPORTANT — do this every time you open a new terminal)
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

> 💡 **In VSCode**: Press `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `./venv/Scripts/python.exe`

---

### STEP 5 — Install Dependencies

With venv activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take 5–10 minutes (TensorFlow is large ~500MB).

> ⚠️ If pip install fails on `tensorflow`:
> ```bash
> pip install tensorflow==2.13.0 --extra-index-url https://pypi.org/simple/
> ```

> ⚠️ If you get **Microsoft Visual C++ error**:
> Install from: https://aka.ms/vs/17/release/vc_redist.x64.exe

---

### STEP 6 — Run the App (Demo Mode — No Training Needed)

```bash
python run.py
```

Open your browser: **http://127.0.0.1:8000**

> ⚠️ In Demo Mode, the model uses ImageNet weights (not trained on cracks).
> Results will NOT be accurate. Train the model first for real results.

---

## 🎓 Training the Model

### Download Dataset

**Option A — SDNET2018** (recommended, ~1GB):
1. Download from: https://digitalcommons.usu.edu/all_datasets/48/
2. Extract to any folder (e.g., `C:\data\SDNET2018\`)
3. It has `D\` (decks), `P\` (pavements), `W\` (walls) — each with `C` (cracked) and `U` (uncracked)

**Option B — Concrete Crack Images** (~250MB, easier):
1. Download from: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
2. Extract to any folder (e.g., `C:\data\crack_images\`)
3. Already has `Positive\` (cracked) and `Negative\` (non_cracked)

---

### Organize Dataset

```bash
# For Concrete Crack Images (Kaggle dataset):
python train/prepare_dataset.py --source C:\data\crack_images --split 0.70 0.15

# Check what was organized:
python train/prepare_dataset.py --check
```

---

### Train the Model

```bash
# Basic training (recommended first run):
python train/train_model.py --epochs 15 --batch 16

# With fine-tuning (better accuracy, slower):
python train/train_model.py --epochs 20 --batch 16 --fine_tune

# Low memory laptop (reduce batch size):
python train/train_model.py --epochs 15 --batch 8
```

Training will:
- Print progress per epoch (loss, accuracy)
- Save best model to `models/crack_model.h5` automatically
- Save training plot to `models/training_history.png`

**Expected time**: ~15–40 min on CPU (no GPU) for 15 epochs

---

### Evaluate the Model

```bash
python train/evaluate_model.py
```

Generates in `models/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `sample_predictions.png`
- Prints precision, recall, F1-score, AUC

---

### Run App with Trained Model

```bash
python run.py
```

The app now uses your trained model automatically.

---

## 🚀 Using the App

Open **http://127.0.0.1:8000** in Chrome or Firefox.

| Button | What it does |
|--------|-------------|
| 📷 Live Camera | Opens webcam for real-time analysis via WebSocket |
| 🖼 Upload Image | Upload a JPG/PNG of a wall, beam, column, etc. |
| 🎬 Upload Video | Upload a video — analyzes every 30th frame, shows worst case |
| ■ Stop | Stops camera and WebSocket |

### Understanding the Output:

- **Grad-CAM heatmap overlay** — Red/warm areas = where model detected damage
- **Severity score** — 0–100% damage confidence
- **Damage zones** — Bounding boxes on specific damaged patches
- **Recommended Actions** — Civil engineering repair steps
- **Activity log** — Real-time events and alerts

---

## 🛠️ Troubleshooting (Windows)

| Problem | Fix |
|---------|-----|
| `venv\Scripts\activate` fails | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell |
| `ModuleNotFoundError` | Make sure venv is activated (see `(venv)` in terminal) |
| TensorFlow import error | Reinstall: `pip install tensorflow==2.13.0` |
| Camera not working in browser | Use Chrome/Firefox, allow camera permission in browser |
| Port 8000 already in use | Run: `python run.py --port 8080` |
| Out of memory during training | Reduce batch: `--batch 8` or even `--batch 4` |
| Very slow training | Normal on CPU — reduce epochs or use `--batch 8` |

---

## 📊 Expected Model Performance (After Training)

| Metric | Expected |
|--------|---------|
| Accuracy | 92–97% |
| Precision | 90–96% |
| Recall | 91–95% |
| F1-Score | 91–95% |
| AUC | 0.97–0.99 |

*(Depends on dataset quality and training duration)*

---

## 🔁 Daily Workflow (Opening the Project Again)

```bash
# 1. Open VSCode → Open Folder → structural-damage-detector
# 2. Open terminal (Ctrl+`)
# 3. Activate venv:
venv\Scripts\activate
# 4. Start app:
python run.py
# 5. Open browser: http://127.0.0.1:8000
```

---

## 📌 Tech Stack Summary

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI + Uvicorn |
| Live Streaming | WebSocket |
| ML Model | MobileNetV2 (TensorFlow/Keras) |
| Visualization | Grad-CAM + OpenCV |
| Localization | Sliding Window Detection |
| Alerts | Rule-based Severity Engine |
| Frontend | Vanilla HTML/CSS/JS |

---

*Project: Early Structural Damage Detection — B.Tech CV/ML Project*
