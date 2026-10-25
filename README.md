# GestureFlow AI 🖐️⚡
> **Real-Time Hand Gesture Recognition & Human-Computer Interface (HCI)**  
> Built with Python, OpenCV, MediaPipe, scikit-learn, and Streamlit.

---

## 📌 Project Overview
**GestureFlow AI** is a production-grade, modular computer vision and human-computer interaction (HCI) application designed to translate real-time physical hand gestures into seamless, safe operating system controls (e.g., cursor navigation, virtual clicks, volume adjustment, media playback, presentation navigation).

---

## 🏗️ System Architecture & Engineering Concepts (Placement Ready)

### 1. Two-Stage Vision Pipeline (MediaPipe BlazePalm + Landmark Regression)
* **Interview Insight**: Why not run a deep landmark detection neural network across the full 1080p frame every frame?
  - A full-frame 3D coordinate regressor is computationally expensive and struggles with scale/rotation variance (e.g., small hand far from camera vs large hand up close).
  - **Solution**: MediaPipe employs a lightweight **BlazePalm detector** that first identifies the rigid palm bounding box (rotation/scale invariant). Then, a specialized **Hand Landmark Model** crops the ROI (Region of Interest) and outputs **21 3D landmarks** with $(x, y, z)$ coordinates.
  - **Tracking Optimization**: Once hand coordinates are localized, subsequent frames bypass the palm detector and track landmarks directly across frames until tracking confidence drops below `min_tracking_confidence`. This maintains 30+ FPS on modest CPU hardware.

### 2. Coordinate System Transformations
* **Normalized Space ($[0.0, 1.0]$)**: Scale and resolution independent coordinates relative to frame width and height.
* **Pixel Space ($[0, W], [0, H]$)**: Multiplied by actual camera resolution $(W, H)$ with the origin $(0,0)$ at the **top-left corner**.
* **Key Heuristic**: In screen coordinates, moving **upwards** corresponds to a **decreasing $y$ coordinate** ($y_{\text{tip}} < y_{\text{pip}}$ signifies an extended finger).

### 3. Fail-Safe Human-Computer Interaction
* OS-level controllers (like `pyautogui`) must implement:
  - Hardware fail-safes (`pyautogui.FAILSAFE = True`, slamming cursor into screen corner aborts).
  - Deadzones & smoothing (Exponential Moving Average / Kalman filtering) to eliminate hand jitter.
  - State debouncing (cooldown timers) to prevent rapid multi-clicks or accidental trigger loops.

---

## 🗺️ Implementation Roadmap

| Phase | Milestone | Status | Key Deliverables |
|---|---|---|---|
| **Phase 1** | **Environment & Vision Foundation** | ✅ **Ready for Verification** | Environment diagnostics, `HandDetector` module, modular config, webcam & landmark test |
| **Phase 2** | **Feature Engineering & Gesture Classification** | ⏳ *Next Step* | Angle/distance vectorization, rule-based & scikit-learn ML gesture classifier (SVM/Random Forest) |
| **Phase 3** | **Safe OS Controls & HCI Engine** | ⏳ *Upcoming* | Smooth cursor movement, virtual click, volume control, emergency fail-safe triggers |
| **Phase 4** | **Model Evaluation & Placement Metrics** | ⏳ *Upcoming* | Confusion matrix, Precision/Recall, latency profiling (inference vs rendering) |
| **Phase 5** | **Interactive Streamlit Dashboard** | ⏳ *Upcoming* | Real-time parameter tuning, gesture confidence monitors, live HUD web interface |

---

## 🚀 Quickstart & Verification

### Step 1: Check Environment Diagnostics
Run the diagnostic script:
```powershell
python tests/test_imports.py
```

### Step 2: Install Remaining Core Vision Packages
> [!NOTE]
> Since Windows Drive `C:\` has limited free storage, redirect pip download cache to Drive `D:\` (which has over 140 GB free):
```powershell
pip install --cache-dir D:\pip_cache opencv-python mediapipe pyautogui
```

### Step 3: Run Geometry Unit Tests
Verify mathematical calculations and coordinate logic:
```powershell
python tests/test_geometry.py
```

### Step 4: Run Real-Time Webcam & Hand Landmark Test
Launch the interactive camera test:
```powershell
python run_webcam_test.py
```
- Hold your hand in front of the camera to verify the 21-point skeletal mesh.
- Pinch your Thumb and Index finger together to test distance-based pinch detection.
- Press **`q`** to safely close the camera feed.
