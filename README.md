
# GestureFlow AI 🖐️⚡
### Real-Time Hand Gesture Recognition & Human-Computer Interaction

GestureFlow AI is a Python-based computer vision project that recognizes hand gestures in real time and uses them to interact with a computer. It combines webcam input, hand landmark detection, gesture classification, and desktop automation to create a touch-free human-computer interaction experience.

**Built with:** Python · OpenCV · MediaPipe · scikit-learn · PyAutoGUI · Streamlit

---

## ✨ Features

- **Real-Time Hand Tracking:** Detects and tracks hand landmarks using MediaPipe.
- **Gesture Recognition:** Recognizes gestures such as Fist, Open Palm, Peace Sign, Pointing, Thumbs Up, and Pinch.
- **Touch-Free Interaction:** Supports cursor movement, click actions, and swipe-based scrolling.
- **Modular Architecture:** Separates configuration, hand detection, gesture recognition, and webcam controls.
- **Live Visual Feedback:** Displays recognized gestures and tracking status through an OpenCV window.
- **Testing & Diagnostics:** Includes scripts for checking dependencies, camera access, and hand landmark detection.

---

## 🏗️ System Architecture

GestureFlow AI follows a modular computer vision pipeline:

```text
Webcam Input
     │
     ▼
Frame Preprocessing
     │
     ▼
Hand Detection & Landmark Tracking
        (MediaPipe)
     │
     ▼
Gesture Recognition
     │
     ▼
Gesture-to-Action Mapping
     │
     ▼
Computer Controls
(Cursor · Click · Scroll)
```

### 1. Hand Detection and Landmark Tracking

MediaPipe identifies the hand and estimates **21 landmarks** representing key points on the fingers, joints, and palm.

The landmark coordinates provide the geometric information needed to identify finger positions and distinguish between gestures.

### 2. Coordinate System

Hand landmarks use normalized coordinates, where `x` and `y` generally range from `0.0` to `1.0`.

- **Normalized coordinates:** Represent positions relative to the camera frame.
- **Pixel coordinates:** Convert normalized positions using the frame width and height.
- **Finger extension:** Vertical landmark positions can help identify extended fingers when the hand is oriented appropriately.

### 3. Gesture Recognition

The gesture recognition module analyzes hand landmarks and applies geometric rules to classify gestures.

The current implementation uses rule-based recognition. A machine-learning classifier, such as SVM or Random Forest, can be explored as a future enhancement.

### 4. Safe Computer Interaction

PyAutoGUI translates recognized gestures into desktop actions.

The interaction layer can be extended with cursor smoothing, gesture debouncing, cooldowns, and emergency stop mechanisms to improve reliability and prevent accidental actions.

---

## 🖐️ Supported Gestures

| Gesture | Action |
|---|---|
| Pointing | Move the cursor |
| Pinch | Perform a click |
| Open Palm + swipe up | Scroll down |
| Open Palm + swipe down | Scroll up |
| Fist | Recognize a closed hand |
| Peace Sign | Recognize a two-finger gesture |
| Thumbs Up | Recognize a thumbs-up gesture |

*Computer controls must be enabled before performing desktop actions.*

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| OpenCV | Webcam access, image processing, and visual feedback |
| MediaPipe | Hand detection and landmark tracking |
| scikit-learn | Potential future ML-based gesture classification |
| PyAutoGUI | Desktop automation and input control |
| Streamlit | Planned interactive dashboard |

---

## 📂 Project Structure

```text
GestureFlowAI/
├── data/
├── models/
│   └── hand_landmarker.task
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── hand_detector.py
│   └── gesture_recognizer.py
├── tests/
│   ├── __init__.py
│   ├── diagnose_inference.py
│   ├── test_camera_backend.py
│   ├── test_geometry.py
│   ├── test_gui_window.py
│   ├── test_hand_landmarker.py
│   ├── test_imports.py
│   └── test_webcam.py
├── .gitignore
├── requirements.txt
├── run_webcam_test.py
├── test_opencv_window.py
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python installed on your system
- A working webcam
- Git
- A Windows, Linux, or macOS environment compatible with the required libraries

### 1. Clone the Repository

```bash
git clone https://github.com/jyothireddy2106/GestureFlowAI.git
cd GestureFlowAI
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the Python executable directly from the virtual environment instead.

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need to keep pip's download cache on another drive, use:

```powershell
pip install --cache-dir D:\pip_cache -r requirements.txt
```

### 4. Verify the Environment

```bash
python tests/test_imports.py
```

### 5. Run Geometry Tests

```bash
python tests/test_geometry.py
```

### 6. Launch GestureFlow AI

```bash
python run_webcam_test.py
```

Allow webcam access when prompted. Hold your hand in front of the camera to view the detected landmarks and recognized gestures.

---

## ⌨️ Keyboard Controls

| Key | Function |
|---|---|
| `C` | Enable or disable computer controls |
| `T` | Toggle hand tracking |
| `S` | Save a snapshot |
| `Q` | Quit the application |

**Tip:** Enable controls with `C` before testing cursor movement, clicking, or scrolling. Position your cursor over the application or webpage you want to control.

---

## 🧪 Testing and Diagnostics

The project includes scripts for testing different components:

- `tests/test_imports.py` — Check dependency imports.
- `tests/test_geometry.py` — Test geometric calculations.
- `tests/test_hand_landmarker.py` — Test hand landmark detection.
- `tests/test_camera_backend.py` — Check camera backend behavior.
- `tests/test_gui_window.py` — Test window behavior.
- `tests/diagnose_inference.py` — Diagnose inference issues.

Run individual tests using Python from the project root.

---

## 🗺️ Development Roadmap

- [x] Set up the Python environment and core vision pipeline
- [x] Implement hand detection and landmark tracking
- [x] Implement rule-based gesture recognition
- [x] Add cursor movement and click controls
- [x] Implement swipe-based scrolling
- [ ] Collect gesture datasets for ML-based classification
- [ ] Evaluate classifiers using precision, recall, and a confusion matrix
- [ ] Measure inference latency and frame rate
- [ ] Add advanced desktop controls, such as volume and media playback
- [ ] Build an interactive Streamlit dashboard
- [ ] Improve gesture calibration, accessibility, and safety controls

---

## 🔮 Future Enhancements

- Machine-learning-based gesture classification
- Personalized gesture calibration
- Advanced cursor smoothing and jitter reduction
- Volume and media playback controls
- Gesture confidence visualization
- Performance metrics and latency profiling
- An interactive Streamlit dashboard

---

## 🎯 Project Goals

GestureFlow AI explores how computer vision and gesture recognition can make human-computer interaction more natural and accessible. It also provides a practical foundation for learning computer vision, geometric feature extraction, software modularity, testing, and desktop automation.

---

## 👩‍💻 Author

**Jyothi Reddy**

- GitHub: [@jyothireddy2106](https://github.com/jyothireddy2106)

---
