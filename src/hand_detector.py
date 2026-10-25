"""
GestureFlow AI - Hand Detector Module
Supports both MediaPipe Tasks API (v1.0+) and legacy MediaPipe Solutions API (v0.10).
Preserves Clean Architecture, high-framerate processing, and student placement concepts.
"""

import time
import math
import os
from typing import List, Tuple, Dict, Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

from src.config import MediaPipeConfig, HandLandmarks, UIColors, resolve_model_path


class HandDetector:
    """Modular Hand Detector and Landmark Tracker.
    
    Compatible with:
      - MediaPipe 1.0+ (Tasks HandLandmarker API using .task bundle)
      - MediaPipe 0.10.x (Legacy Solutions Hands API)
    
    Placement Concept - Systems Robustness & Thread Safety:
    In MediaPipe Tasks (1.1.1 on Windows), `RunningMode.VIDEO` routes calls
    through an internal C++ thread-pool dispatcher (`serial_dispatcher`) that
    deadlocks due to disabled feedback tensors in `InferenceFeedbackManager`.
    By using `RunningMode.IMAGE` for each captured frame, inference executes as
    a direct, synchronous forward pass in ~9 ms (yielding ~108 FPS capacity),
    guaranteeing zero deadlocks, instant startup, and seamless integration with
    standard OpenCV webcam event loops.
    """

    def __init__(
        self,
        mode: bool = MediaPipeConfig.STATIC_IMAGE_MODE,
        max_hands: int = MediaPipeConfig.MAX_NUM_HANDS,
        model_complexity: int = MediaPipeConfig.MODEL_COMPLEXITY,
        detection_con: float = MediaPipeConfig.MIN_DETECTION_CONFIDENCE,
        track_con: float = MediaPipeConfig.MIN_TRACKING_CONFIDENCE,
        model_path: Optional[str] = None,
        use_video_mode: bool = False,
    ):
        if cv2 is None or mp is None:
            raise ImportError(
                "Missing core vision dependencies ('opencv-python' or 'mediapipe').\n"
                "Please run: pip install opencv-python mediapipe"
            )

        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.use_video_mode = use_video_mode

        # Detect which MediaPipe API is available (Tasks API vs Legacy Solutions)
        self.is_tasks_api = hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")
        self.detector = None
        self.running_mode = None
        self.results: Any = None
        self._last_timestamp_ms: int = 0

        if self.is_tasks_api:
            self._init_tasks_api(model_path)
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._init_legacy_solutions()
        else:
            raise AttributeError(
                "Unsupported MediaPipe installation: neither 'mediapipe.tasks.vision' "
                "nor 'mediapipe.solutions.hands' was found."
            )

        # Performance & FPS tracking
        self.prev_time: float = time.time()
        self.fps: float = 0.0

    def _init_tasks_api(self, model_path: Optional[str] = None):
        """Initialize modern MediaPipe Tasks HandLandmarker."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        resolved_path = model_path if model_path else resolve_model_path()
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Hand landmarker model file not found at: {resolved_path}")

        base_options = python.BaseOptions(model_asset_path=resolved_path)
        
        # Default to IMAGE mode for synchronous, deadlock-free 100+ FPS real-time webcam processing
        self.running_mode = vision.RunningMode.VIDEO if self.use_video_mode else vision.RunningMode.IMAGE
        self.is_video_mode = (self.running_mode == vision.RunningMode.VIDEO)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_con,
            min_hand_presence_confidence=MediaPipeConfig.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.track_con,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _init_legacy_solutions(self):
        """Initialize legacy MediaPipe Solutions Hands."""
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )

    def find_hands(self, img: "np.ndarray", draw: bool = True) -> "np.ndarray":
        """Process frame, execute hand detection/tracking, and render landmarks.
        
        Args:
            img: BGR image frame from OpenCV.
            draw: Whether to render landmark points and bone connections.
            
        Returns:
            Annotated BGR image frame.
        """
        if img is None:
            raise ValueError("Input image frame cannot be None.")

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        if self.is_tasks_api:
            # Modern MediaPipe Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            if getattr(self, "is_video_mode", False):
                current_ms = int(time.time() * 1000)
                timestamp_ms = max(current_ms, self._last_timestamp_ms + 1)
                self._last_timestamp_ms = timestamp_ms
                self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
            else:
                # Synchronous, reliable forward pass (9-10 ms per frame)
                self.results = self.detector.detect(mp_image)

            # Draw landmarks directly via high-performance OpenCV rendering
            if draw and self.results and self.results.hand_landmarks:
                for hand_lms in self.results.hand_landmarks:
                    self._draw_hand_skeleton(img, hand_lms, w, h)
        else:
            # Legacy Solutions API
            img_rgb.flags.writeable = False
            self.results = self.detector.process(img_rgb)
            img_rgb.flags.writeable = True

            if draw and self.results and self.results.multi_hand_landmarks:
                for hand_lms in self.results.multi_hand_landmarks:
                    self._draw_hand_skeleton(img, hand_lms.landmark, w, h)

        return img

    def _draw_hand_skeleton(self, img: "np.ndarray", landmarks: Any, w: int, h: int):
        """Render high-contrast bone connections and joint circles."""
        # Convert landmarks to pixel coordinates
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Draw bone connections
        for p1_idx, p2_idx in HandLandmarks.CONNECTIONS:
            if p1_idx < len(pts) and p2_idx < len(pts):
                cv2.line(img, pts[p1_idx], pts[p2_idx], UIColors.CYAN_BLUE, 2, cv2.LINE_AA)

        # Draw joints
        for i, (px, py) in enumerate(pts):
            if i in HandLandmarks.FINGER_TIPS:
                # Fingertips highlighted in alert red/accent
                cv2.circle(img, (px, py), 6, UIColors.ALERT_RED, cv2.FILLED, cv2.LINE_AA)
            else:
                # Knuckles and intermediate joints in emerald green
                cv2.circle(img, (px, py), 4, UIColors.EMERALD_GREEN, cv2.FILLED, cv2.LINE_AA)

    def find_positions(
        self, img: "np.ndarray", hand_idx: int = 0, draw_bbox: bool = True
    ) -> Tuple[List[Dict[str, Any]], Tuple[int, int, int, int]]:
        """Extract coordinate data for all 21 landmarks of a specific hand.
        
        Args:
            img: BGR image frame.
            hand_idx: Index of the detected hand (0 for primary hand).
            draw_bbox: Whether to draw a bounding rectangle around the hand.
            
        Returns:
            Tuple of:
              - landmark_list: List of dicts with 'id', 'px', 'py', 'norm_x', 'norm_y', 'norm_z'.
              - bounding_box: (xmin, ymin, xmax, ymax) in pixel coordinates.
        """
        lm_list: List[Dict[str, Any]] = []
        bbox = (0, 0, 0, 0)

        if not self.results:
            return lm_list, bbox

        # Extract landmark list based on active API
        if self.is_tasks_api:
            if not self.results.hand_landmarks or hand_idx >= len(self.results.hand_landmarks):
                return lm_list, bbox
            landmarks = self.results.hand_landmarks[hand_idx]
        else:
            if not self.results.multi_hand_landmarks or hand_idx >= len(self.results.multi_hand_landmarks):
                return lm_list, bbox
            landmarks = self.results.multi_hand_landmarks[hand_idx].landmark

        h, w, _ = img.shape
        x_coords: List[int] = []
        y_coords: List[int] = []

        for lm_id, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            x_coords.append(px)
            y_coords.append(py)

            lm_list.append({
                "id": lm_id,
                "px": px,
                "py": py,
                "norm_x": float(lm.x),
                "norm_y": float(lm.y),
                "norm_z": float(lm.z),
            })

        if x_coords and y_coords:
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            # Add padding to bounding box
            padding = 15
            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(w, xmax + padding)
            ymax = min(h, ymax + padding)
            bbox = (xmin, ymin, xmax, ymax)

            if draw_bbox:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), UIColors.EMERALD_GREEN, 2)

        return lm_list, bbox

    def get_handedness(self, hand_idx: int = 0) -> str:
        """Returns 'Left' or 'Right' classification for the detected hand."""
        if not self.results:
            return "Unknown"

        if self.is_tasks_api:
            if not self.results.handedness or hand_idx >= len(self.results.handedness):
                return "Unknown"
            category = self.results.handedness[hand_idx][0]
            return category.category_name or category.display_name or "Unknown"
        else:
            if not self.results.multi_handedness or hand_idx >= len(self.results.multi_handedness):
                return "Unknown"
            return self.results.multi_handedness[hand_idx].classification[0].label

    def fingers_up(self, lm_list: List[Dict[str, Any]], handedness: str = "Right") -> List[int]:
        """Determine which fingers are extended (up = 1, down = 0).
        
        Args:
            lm_list: List of 21 landmark dictionaries.
            handedness: 'Right' or 'Left' to resolve horizontal thumb extension.
            
        Returns:
            List of 5 binary indicators: [Thumb, Index, Middle, Ring, Pinky]
            
        Interview Note - Geometry vs Heuristics:
        In standard 2D image coordinates:
        - Origin (0,0) is TOP-LEFT.
        - Higher on the screen means SMALLER Y value.
        - Therefore, Finger is extended if: tip.y < pip.y
        - For Thumb, it articulates horizontally (X-axis).
        """
        if len(lm_list) < 21:
            return [0, 0, 0, 0, 0]

        fingers = [0, 0, 0, 0, 0]

        # 1. Thumb horizontal extension check
        thumb_tip_x = lm_list[HandLandmarks.THUMB_TIP]["px"]
        thumb_ip_x = lm_list[HandLandmarks.THUMB_IP]["px"]
        if handedness == "Right":
            fingers[0] = 1 if thumb_tip_x > thumb_ip_x else 0
        else:
            fingers[0] = 1 if thumb_tip_x < thumb_ip_x else 0

        # 2. 4 Fingers vertical extension check (Index, Middle, Ring, Pinky)
        finger_tips = [
            HandLandmarks.INDEX_FINGER_TIP,
            HandLandmarks.MIDDLE_FINGER_TIP,
            HandLandmarks.RING_FINGER_TIP,
            HandLandmarks.PINKY_TIP,
        ]
        finger_pips = [
            HandLandmarks.INDEX_FINGER_PIP,
            HandLandmarks.MIDDLE_FINGER_PIP,
            HandLandmarks.RING_FINGER_PIP,
            HandLandmarks.PINKY_PIP,
        ]

        for i in range(4):
            tip_y = lm_list[finger_tips[i]]["py"]
            pip_y = lm_list[finger_pips[i]]["py"]
            fingers[i + 1] = 1 if tip_y < pip_y else 0

        return fingers

    def calculate_distance(
        self,
        p1_id: int,
        p2_id: int,
        lm_list: List[Dict[str, Any]],
        img: Optional["np.ndarray"] = None,
        draw: bool = True,
    ) -> Tuple[float, Optional["np.ndarray"], Tuple[int, int]]:
        """Calculate Euclidean distance between two landmark points."""
        if len(lm_list) < 21:
            return 0.0, img, (0, 0)

        x1, y1 = lm_list[p1_id]["px"], lm_list[p1_id]["py"]
        x2, y2 = lm_list[p2_id]["px"], lm_list[p2_id]["py"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distance = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.circle(img, (x1, y1), 8, UIColors.PURPLE, cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, UIColors.PURPLE, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), UIColors.PURPLE, 2)
            cv2.circle(img, (cx, cy), 6, UIColors.ALERT_RED, cv2.FILLED)

        return distance, img, (cx, cy)

    def update_fps(self) -> float:
        """Calculate real-time running FPS using frame delta time."""
        curr_time = time.time()
        dt = curr_time - self.prev_time
        self.fps = (1.0 / dt) if dt > 0 else 0.0
        self.prev_time = curr_time
        return self.fps

    def close(self):
        """Release underlying model resources."""
        if self.detector is not None:
            if hasattr(self.detector, "close"):
                self.detector.close()
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
