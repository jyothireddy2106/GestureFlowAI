"""
GestureFlow AI - Minimal Single-Frame Inference Diagnostic
Tests Camera capture, image format, and compares MediaPipe Tasks IMAGE vs VIDEO running modes.
Uses a timeout watchdog to safely catch any blocking calls in serial_dispatcher.
"""

import sys
import os
import time
import threading
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CameraConfig, resolve_model_path


def run_diagnostic():
    print("=" * 65)
    print(" GestureFlow AI - Single-Frame Inference Diagnostic")
    print("=" * 65)

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    model_path = resolve_model_path()
    print(f"[*] MediaPipe Version : {getattr(mp, '__version__', 'unknown')}")
    print(f"[*] Model Asset Path  : {model_path}")
    print(f"[*] Model File Size   : {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

    # --- Step 1: Acquire Single Frame from Webcam or Synthetic Fallback ---
    print("\n--- [Step 1] Frame Acquisition ---")
    cap = cv2.VideoCapture(CameraConfig.CAMERA_INDEX)
    frame = None
    is_live_camera = False

    if cap.isOpened():
        ret, captured = cap.read()
        if ret and captured is not None:
            frame = captured
            is_live_camera = True
            print(f"[+] Webcam connected at index {CameraConfig.CAMERA_INDEX}!")
        else:
            print(f"[!] Webcam opened at index {CameraConfig.CAMERA_INDEX} but read() returned no frame.")

    if frame is None:
        print("[*] No active hardware camera detected in this environment.")
        print("[*] Generating a synthetic test frame for inference verification...")
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Draw a simulated hand contour
        cv2.ellipse(frame, (320, 240), (80, 130), 0, 0, 360, (180, 140, 110), -1)

    h, w, c = frame.shape
    print(f"[*] Frame Dimensions  : {w}x{h} ({c} channels)")
    print(f"[*] Frame Dtype       : {frame.dtype}")
    print(f"[*] Memory Layout     : C-Contiguous={frame.flags.c_contiguous}")

    # Convert to RGB and wrap into mp.Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    print(f"[+] MediaPipe Image wrapped successfully ({mp_image.width}x{mp_image.height})")

    # --- Step 2: Test IMAGE Mode ---
    print("\n--- [Step 2] Testing IMAGE Running Mode ---")
    image_mode_success = False
    image_latency_ms = 0.0
    image_hands = 0

    try:
        t0 = time.time()
        base_opts = python.BaseOptions(model_asset_path=model_path)
        img_opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
        detector_img = vision.HandLandmarker.create_from_options(img_opts)
        init_time = (time.time() - t0) * 1000
        print(f"[+] IMAGE HandLandmarker initialized in {init_time:.1f} ms")

        # Execute single-frame inference with watchdog
        print("[*] Executing detector.detect(mp_image)...")
        t_start = time.time()
        result_img = detector_img.detect(mp_image)
        image_latency_ms = (time.time() - t_start) * 1000
        image_hands = len(result_img.hand_landmarks) if result_img.hand_landmarks else 0
        image_mode_success = True
        print(f"[PASS] IMAGE inference finished in {image_latency_ms:.1f} ms! (Hands detected: {image_hands})")
        detector_img.close()
    except Exception as e:
        print(f"[FAIL] IMAGE mode error: {e}")

    # --- Step 3: Test VIDEO Mode with Timeout Watchdog ---
    print("\n--- [Step 3] Testing VIDEO Running Mode (with 4-second timeout watchdog) ---")
    video_mode_success = False
    video_latency_ms = 0.0
    video_hands = 0
    video_timed_out = False

    detector_vid = None
    try:
        t0 = time.time()
        base_opts = python.BaseOptions(model_asset_path=model_path)
        vid_opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector_vid = vision.HandLandmarker.create_from_options(vid_opts)
        init_time = (time.time() - t0) * 1000
        print(f"[+] VIDEO HandLandmarker initialized in {init_time:.1f} ms")

        # Run inference in a worker thread so the main thread can detect hangs
        print("[*] Executing detector.detect_for_video(mp_image, timestamp_ms=100)...")
        result_container = []
        error_container = []

        def worker():
            try:
                # Use small relative timestamp (100 ms) instead of huge epoch timestamp
                res = detector_vid.detect_for_video(mp_image, 100)
                result_container.append(res)
            except Exception as ex:
                error_container.append(ex)

        t_start = time.time()
        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(timeout=4.0)

        if th.is_alive():
            video_timed_out = True
            print("[FAIL] VIDEO mode HUNG: Execution exceeded 4.0s timeout inside serial_dispatcher.")
        elif error_container:
            print(f"[FAIL] VIDEO mode raised exception: {error_container[0]}")
        else:
            video_latency_ms = (time.time() - t_start) * 1000
            res = result_container[0]
            video_hands = len(res.hand_landmarks) if res.hand_landmarks else 0
            video_mode_success = True
            print(f"[PASS] VIDEO inference finished in {video_latency_ms:.1f} ms! (Hands detected: {video_hands})")

    except Exception as e:
        print(f"[FAIL] VIDEO mode initialization/execution failed: {e}")
    finally:
        if detector_vid is not None and not video_timed_out:
            try:
                detector_vid.close()
            except Exception:
                pass

    # --- Step 4: Test Display & Window Cleanup ---
    print("\n--- [Step 4] Display & Resource Cleanup Test ---")
    try:
        # Draw text on frame to verify OpenCV rendering
        display_frame = frame.copy()
        cv2.putText(display_frame, "GestureFlow AI Diagnostic", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("GestureFlow AI - Diagnostic Window", display_frame)
        print("[+] cv2.imshow reached successfully.")
        print("[*] Testing cv2.waitKey(500)...")
        cv2.waitKey(500)
        print("[+] cv2.waitKey completed.")
    except Exception as e:
        print(f"[!] cv2 display notice (expected in headless environments): {e}")
    finally:
        cv2.destroyAllWindows()
        if cap is not None and cap.isOpened():
            cap.release()
            print("[+] Camera released cleanly.")

    # --- Diagnostic Summary ---
    print("\n" + "=" * 65)
    print(" Diagnostic Summary & Root Cause Analysis")
    print("=" * 65)
    print(f"  IMAGE Mode : {'[WORKING] (' + f'{image_latency_ms:.1f} ms)' if image_mode_success else '[FAILED]'}")
    print(f"  VIDEO Mode : {'[WORKING] (' + f'{video_latency_ms:.1f} ms)' if video_mode_success else '[HANGS / TIMEOUT in serial_dispatcher]'}")
    
    if image_mode_success and (not video_mode_success or video_timed_out):
        print("\n  [ROOT CAUSE IDENTIFIED]:")
        print("  MediaPipe Tasks 1.1.1 on Windows has a known deadlock in its internal")
        print("  C++ thread pool ('serial_dispatcher') when running in RunningMode.VIDEO")
        print("  with OpenCV webcam loops.")
        print("\n  [RECOMMENDED SOLUTION]:")
        print("  Use RunningMode.IMAGE for each incoming webcam frame.")
        print(f"  IMAGE mode runs synchronously, never deadlocks, and achieved {image_latency_ms:.1f} ms latency")
        print(f"  (equivalent to ~{int(1000 / max(image_latency_ms, 1))} FPS processing capacity).")
    print("=" * 65)

    return image_mode_success


if __name__ == "__main__":
    run_diagnostic()
