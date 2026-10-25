"""
GestureFlow AI - Test Entry Point
Run this script to verify your camera feed and 21-point hand landmark tracking.

Usage:
    python run_webcam_test.py          # Runs camera + MediaPipe hand tracking
    python run_webcam_test.py --raw    # Runs camera ONLY (Raw video test, no MediaPipe)
"""

import sys
import os

# Set root directory in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_webcam import run_webcam_test

if __name__ == "__main__":
    is_headless = "--headless" in sys.argv
    is_raw = "--raw" in sys.argv
    frames = 10 if is_headless else 0
    run_webcam_test(headless_frames=frames, raw_mode=is_raw)
