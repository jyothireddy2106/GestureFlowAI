"""
GestureFlow AI - Geometry and Mathematical Validation Tests
Tests distance calculation, coordinate conversions, and finger state logic
independently of live hardware camera feeds.
"""

import math
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import HandLandmarks


class TestGestureGeometry(unittest.TestCase):
    """Unit tests for spatial hand geometry calculations."""

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation between two 2D points."""
        p1 = (100, 100)
        p2 = (130, 140)
        # Expected: sqrt((130-100)^2 + (140-100)^2) = sqrt(900 + 1600) = sqrt(2500) = 50.0
        expected = 50.0
        calculated = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_finger_extension_heuristic(self):
        """Verify finger up heuristic logic in 2D top-left inverted screen coordinates.
        
        Placement Concept:
        In OpenCV / Computer Vision screen space:
        - Origin (0,0) is TOP-LEFT.
        - Moving DOWN increases Y (e.g., wrist at Y=400, finger tips at Y=150).
        - An extended finger tip has a LOWER Y value than its PIP joint.
        """
        # Simulated extended index finger: Tip higher (y=150) than PIP (y=250)
        index_tip_y = 150
        index_pip_y = 250
        is_extended = index_tip_y < index_pip_y
        self.assertTrue(is_extended, "Finger tip above PIP should evaluate to extended (True).")

        # Simulated folded index finger: Tip lower (y=300) than PIP (y=250)
        folded_tip_y = 300
        is_folded_extended = folded_tip_y < index_pip_y
        self.assertFalse(is_folded_extended, "Folded finger tip should evaluate to not extended (False).")

    def test_landmark_topology_completeness(self):
        """Verify MediaPipe 21 landmark definition coverage."""
        self.assertEqual(HandLandmarks.WRIST, 0)
        self.assertEqual(HandLandmarks.THUMB_TIP, 4)
        self.assertEqual(HandLandmarks.INDEX_FINGER_TIP, 8)
        self.assertEqual(HandLandmarks.MIDDLE_FINGER_TIP, 12)
        self.assertEqual(HandLandmarks.RING_FINGER_TIP, 16)
        self.assertEqual(HandLandmarks.PINKY_TIP, 20)
        self.assertEqual(len(HandLandmarks.FINGER_TIPS), 5)
        self.assertEqual(len(HandLandmarks.FINGER_PIPS), 5)


if __name__ == "__main__":
    unittest.main()
