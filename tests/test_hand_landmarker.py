"""
GestureFlow AI - Hand Landmarker & Tasks API Unit Test
Validates that HandDetector initializes with MediaPipe Tasks and runs inference cleanly.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hand_detector import HandDetector
from src.config import HandLandmarks, resolve_model_path


class TestHandDetectorTasks(unittest.TestCase):
    """Test HandDetector implementation under modern MediaPipe Tasks API."""

    def test_model_file_exists(self):
        """Verify the hand_landmarker.task model file is resolved."""
        model_path = resolve_model_path()
        self.assertTrue(os.path.isfile(model_path), f"Model file missing: {model_path}")
        self.assertGreater(os.path.getsize(model_path), 5 * 1024 * 1024, "Model file seems incomplete or corrupt.")

    def test_detector_initialization_and_inference(self):
        """Test video-mode hand detector creation and frame processing."""
        with HandDetector(mode=False, max_hands=2) as detector:
            self.assertTrue(detector.is_tasks_api, "Should use MediaPipe Tasks API in mediapipe >= 1.0.")
            self.assertIsNotNone(detector.detector)

            # Generate synthetic test frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            processed = detector.find_hands(dummy_frame, draw=True)
            self.assertEqual(processed.shape, (480, 640, 3))

            # Position extraction on empty frame should return empty lists without crashing
            lm_list, bbox = detector.find_positions(processed, hand_idx=0)
            self.assertEqual(lm_list, [])
            self.assertEqual(bbox, (0, 0, 0, 0))

            # Handedness on empty detection
            handedness = detector.get_handedness(0)
            self.assertEqual(handedness, "Unknown")

            # FPS measurement
            fps = detector.update_fps()
            self.assertGreaterEqual(fps, 0.0)

    def test_synthetic_hand_data_parsing(self):
        """Test finger state heuristics on simulated landmarks."""
        with HandDetector(mode=False, max_hands=1) as detector:
            # Build 21 simulated landmarks with all 5 fingers extended upwards
            simulated_lms = []
            for i in range(21):
                simulated_lms.append({
                    "id": i,
                    "px": 100 + i * 5,
                    # Tips higher on screen (lower y) than PIPs
                    "py": 100 if i in HandLandmarks.FINGER_TIPS else 200,
                    "norm_x": 0.5,
                    "norm_y": 0.5,
                    "norm_z": 0.0,
                })

            # For right hand, thumb tip x > ip x is extended
            simulated_lms[HandLandmarks.THUMB_TIP]["px"] = 250
            simulated_lms[HandLandmarks.THUMB_IP]["px"] = 200

            fingers = detector.fingers_up(simulated_lms, handedness="Right")
            self.assertEqual(fingers, [1, 1, 1, 1, 1], "All 5 fingers should evaluate as open.")

            # Test pinch distance between thumb (4) and index (8)
            dist, _, center = detector.calculate_distance(
                HandLandmarks.THUMB_TIP,
                HandLandmarks.INDEX_FINGER_TIP,
                simulated_lms
            )
            self.assertGreater(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
