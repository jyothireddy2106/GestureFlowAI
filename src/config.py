"""
Configuration settings and constants for GestureFlow AI.
Designed for maintainability, modularity, and interview-ready architectural clarity.
Supports both modern MediaPipe Tasks (v1.0+) and legacy MediaPipe Solutions APIs.
"""

import os
from dataclasses import dataclass
from typing import Tuple, List

# ---------------------------------------------------------
# Placement Concept Note: Configuration Separation
# Keeping hyperparameters, UI constants, and system thresholds
# in a centralized config decouples business logic from tuning.
# ---------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    """Webcam feed configuration parameters.
    
    Interview Note:
    Standard webcams stream at 30 FPS. Processing at 640x480 gives
    optimal trade-off between spatial resolution for small fingers
    and low matrix-multiplication latency on CPU without GPU acceleration.
    On Windows, DirectShow (CAP_DSHOW) is preferred over Media Foundation (CAP_MSMF)
    to prevent IMFSourceReader deadlocks during continuous cap.read() calls.
    """
    CAMERA_INDEX: int = 0
    FALLBACK_CAMERA_INDEX: int = 1
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30
    PREFERRED_BACKEND: str = "DSHOW"


@dataclass(frozen=True)
class MediaPipeConfig:
    """MediaPipe Hand tracking parameters.
    
    Interview Note (How MediaPipe Tasks Works in MediaPipe 1.0+):
    Google restructured MediaPipe in 2024/2025 into 'MediaPipe Tasks'.
    The legacy `mp.solutions.hands` API was replaced by `HandLandmarker`,
    which runs an explicit TensorFlow Lite model bundle (`.task` file).
    
    The pipeline still executes two stages internally:
    1. BlazePalm Detector: Predicts rotated hand bounding boxes.
    2. Hand Landmark Model: Predicts 21 3D coordinates inside the cropped region.
    """
    STATIC_IMAGE_MODE: bool = False       # False = video stream (enables temporal tracking)
    MAX_NUM_HANDS: int = 2                # Support single or dual hands
    MODEL_COMPLEXITY: int = 1             # 0 = Lite (fastest), 1 = Full (balanced)
    MIN_DETECTION_CONFIDENCE: float = 0.7 # Threshold for initial hand detection
    MIN_PRESENCE_CONFIDENCE: float = 0.5  # Threshold for hand presence in tracked region
    MIN_TRACKING_CONFIDENCE: float = 0.5  # Threshold for landmark temporal continuity
    MODEL_FILENAME: str = "hand_landmarker.task"
    MODEL_DOWNLOAD_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )


def resolve_model_path() -> str:
    """Find or download the official MediaPipe hand_landmarker.task model.
    
    Searches:
      1. Local project `models/hand_landmarker.task`
      2. `D:/GestureFlowAI/models/hand_landmarker.task`
      3. Fallback auto-download if connected to network
    """
    # 1. Project-relative directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    local_model = os.path.join(project_root, "models", MediaPipeConfig.MODEL_FILENAME)
    if os.path.isfile(local_model):
        return local_model

    # 2. Hardcoded fallback known project paths
    d_model = os.path.join("D:\\GestureFlowAI", "models", MediaPipeConfig.MODEL_FILENAME)
    if os.path.isfile(d_model):
        return d_model

    # 3. Create models directory and auto-download if missing
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    target_path = os.path.join(models_dir, MediaPipeConfig.MODEL_FILENAME)
    
    if not os.path.isfile(target_path):
        import urllib.request
        print(f"[*] Downloading official MediaPipe Hand Landmarker model to {target_path}...")
        try:
            urllib.request.urlretrieve(MediaPipeConfig.MODEL_DOWNLOAD_URL, target_path)
            print("[+] Model download complete!")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not locate or download hand_landmarker.task model: {e}\n"
                f"Please manually download from:\n{MediaPipeConfig.MODEL_DOWNLOAD_URL}\n"
                f"and place it at: {target_path}"
            )

    return target_path


class HandLandmarks:
    """MediaPipe Hand 21 Landmark Indices Mapping and Skeletal Connections.
    
    Topology:
      - 0: WRIST
      - 1-4: THUMB (CMC, MCP, IP, TIP)
      - 5-8: INDEX (MCP, PIP, DIP, TIP)
      - 9-12: MIDDLE (MCP, PIP, DIP, TIP)
      - 13-16: RING (MCP, PIP, DIP, TIP)
      - 17-20: PINKY (MCP, PIP, DIP, TIP)
    """
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8

    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12

    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [2, 6, 10, 14, 18]

    # Skeletal connections between bone joints for rendering
    CONNECTIONS: List[Tuple[int, int]] = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Pinky finger
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm knuckles / transverse arch
        (5, 9), (9, 13), (13, 17),
    ]


# Color scheme in BGR format (OpenCV standard)
class UIColors:
    EMERALD_GREEN: Tuple[int, int, int] = (46, 204, 113)
    CYAN_BLUE: Tuple[int, int, int] = (255, 191, 0)
    ALERT_RED: Tuple[int, int, int] = (71, 71, 235)
    PURPLE: Tuple[int, int, int] = (204, 50, 153)
    DARK_BG: Tuple[int, int, int] = (30, 30, 30)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    GOLD: Tuple[int, int, int] = (0, 215, 255)
