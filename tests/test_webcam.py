
"""
GestureFlow AI - Webcam, gesture recognition and computer controls.
C - Enable/disable computer controls
T - Toggle hand tracking
S - Save a snapshot
Q - Quit
"""

import sys
import os
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import cv2
import pyautogui

from src.config import CameraConfig, UIColors, HandLandmarks
from src.hand_detector import HandDetector
from src.gesture_recognizer import GestureRecognizer

try:
    import ctypes
    has_win32 = True
except ImportError:
    has_win32 = False


def open_directshow_camera(cam_idx=CameraConfig.CAMERA_INDEX):
    """Open webcam using DirectShow with a Media Foundation fallback."""
    print(f"[DEBUG] Opening camera {cam_idx}...", flush=True)

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)

    if not cap.isOpened():
        cap.release()
        return None, "NONE"

    backend = (
        "CAP_DSHOW"
        if cap.get(cv2.CAP_PROP_BACKEND) == cv2.CAP_DSHOW
        else "CAP_MSMF"
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CameraConfig.TARGET_FPS)

    if backend == "CAP_DSHOW":
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

    return cap, backend


def run_webcam_test(headless_frames=0, raw_mode=False):
    detector = None
    recognizer = None
    cap = None

    try:
        if not raw_mode:
            print("[DEBUG] Initializing hand detector...", flush=True)

            detector = HandDetector(
                mode=False,
                max_hands=2,
                detection_con=0.7,
                track_con=0.5,
            )

            recognizer = GestureRecognizer()

        cap, backend = open_directshow_camera(
            CameraConfig.CAMERA_INDEX
        )

        if cap is None:
            print("[ERROR] Could not open webcam.", flush=True)
            return False

        window = "GestureFlow AI - Live Tracking"

        if headless_frames == 0:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 640, 480)
            cv2.moveWindow(window, 120, 120)

            try:
                cv2.setWindowProperty(
                    window, cv2.WND_PROP_TOPMOST, 1
                )
            except cv2.error:
                pass

        tracking = not raw_mode
        controls_enabled = False
        previous_pinch = False

        # Improved swipe-to-scroll settings.
        swipe_start_y = None
        last_swipe_time = 0.0
        SWIPE_THRESHOLD = 0.10
        SWIPE_COOLDOWN = 0.6
        SCROLL_AMOUNT = 15

        frame_count = 0

        screen_w, screen_h = pyautogui.size()

        # Cursor smoothing.
        smooth_x = None
        smooth_y = None
        smoothing = 0.35

        print("\nControls:", flush=True)
        print("  C - Enable/disable computer controls", flush=True)
        print("  T - Toggle hand tracking", flush=True)
        print("  S - Save snapshot", flush=True)
        print("  Q - Quit", flush=True)
        print("  Open Palm + swipe UP/DOWN - Scroll", flush=True)

        while True:
            success, img = cap.read()

            if not success or img is None:
                print("[ERROR] Could not read webcam frame.", flush=True)
                break

            img = cv2.flip(img, 1)
            frame_count += 1

            gesture = "No Hand Detected"
            pinch_distance = None
            raw_landmarks = None

            if tracking and detector is not None:
                img = detector.find_hands(img, draw=True)

                lm_list, bbox = detector.find_positions(
                    img, hand_idx=0, draw_bbox=True
                )

                if len(lm_list) >= 21 and detector.results is not None:
                    if detector.is_tasks_api:
                        hands = getattr(
                            detector.results,
                            "hand_landmarks",
                            None,
                        )
                    else:
                        hands = getattr(
                            detector.results,
                            "multi_hand_landmarks",
                            None,
                        )

                    if hands:
                        raw_landmarks = hands[0]

                        result = recognizer.recognize(raw_landmarks)
                        gesture = result["gesture"]
                        pinch_distance = result.get("pinch_distance")

                    # Draw the distance line between thumb and index.
                    dist, img, center = detector.calculate_distance(
                        HandLandmarks.THUMB_TIP,
                        HandLandmarks.INDEX_FINGER_TIP,
                        lm_list,
                        img=img,
                        draw=True,
                    )

                    # Show pinch text only when Pinch is recognized.
                    if gesture == "Pinch":
                        cv2.putText(
                            img,
                            "PINCH DETECTED",
                            (center[0] - 50, center[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            UIColors.ALERT_RED,
                            2,
                        )

            pinch_active = gesture == "Pinch"

            # Computer controls only work after pressing C.
            if controls_enabled and raw_landmarks is not None:
                index_tip = raw_landmarks[8]

                # Move the cursor when pointing.
                if gesture == "Pointing":
                    target_x = int(index_tip.x * screen_w)
                    target_y = int(index_tip.y * screen_h)

                    target_x = max(0, min(screen_w - 1, target_x))
                    target_y = max(0, min(screen_h - 1, target_y))

                    if smooth_x is None:
                        smooth_x = target_x
                        smooth_y = target_y
                    else:
                        smooth_x += smoothing * (target_x - smooth_x)
                        smooth_y += smoothing * (target_y - smooth_y)

                    pyautogui.moveTo(
                        int(smooth_x),
                        int(smooth_y),
                    )

                # Click once when Pinch starts.
                if pinch_active and not previous_pinch:
                    pyautogui.click(button="left")

                # Swipe-to-scroll using an Open Palm.
                if gesture == "Open Palm":
                    current_y = index_tip.y

                    if swipe_start_y is None:
                        swipe_start_y = current_y

                    movement = current_y - swipe_start_y
                    now = time.time()

                    if (
                        abs(movement) >= SWIPE_THRESHOLD
                        and now - last_swipe_time >= SWIPE_COOLDOWN
                    ):
                        if movement < 0:
                            # Hand moved UP: webpage scrolls DOWN.
                            pyautogui.scroll(-SCROLL_AMOUNT)
                            print(
                                "Swipe UP detected: scrolling down",
                                flush=True,
                            )
                        else:
                            # Hand moved DOWN: webpage scrolls UP.
                            pyautogui.scroll(SCROLL_AMOUNT)
                            print(
                                "Swipe DOWN detected: scrolling up",
                                flush=True,
                            )

                        last_swipe_time = now
                        swipe_start_y = current_y
                else:
                    # Require Open Palm before another swipe.
                    swipe_start_y = None

            else:
                swipe_start_y = None

            previous_pinch = pinch_active

            # Draw the header.
            overlay = img.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (img.shape[1], 45),
                UIColors.DARK_BG,
                -1,
            )

            cv2.addWeighted(
                overlay, 0.75, img, 0.25, 0, img
            )

            cv2.putText(
                img,
                f"GestureFlow AI [{backend}]",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                UIColors.WHITE,
                2,
            )

            cv2.putText(
                img,
                f"Gesture: {gesture}",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                UIColors.EMERALD_GREEN,
                2,
            )

            if pinch_distance is not None:
                cv2.putText(
                    img,
                    f"Pinch distance: {pinch_distance:.3f}",
                    (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    UIColors.WHITE,
                    2,
                )

            control_status = (
                "CONTROLS ON"
                if controls_enabled
                else "CONTROLS OFF"
            )

            cv2.putText(
                img,
                control_status,
                (15, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                UIColors.ALERT_RED if controls_enabled
                else UIColors.EMERALD_GREEN,
                2,
            )

            cv2.putText(
                img,
                "TRACKING" if tracking else "TRACKING OFF",
                (15, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                UIColors.EMERALD_GREEN,
                2,
            )

            if headless_frames > 0 and frame_count >= headless_frames:
                output_path = os.path.join(
                    os.path.dirname(__file__),
                    "test_snapshot.jpg",
                )

                cv2.imwrite(output_path, img)
                print(f"[+] Snapshot saved: {output_path}", flush=True)
                break

            cv2.imshow(window, img)

            if frame_count == 1 and has_win32:
                hwnd = ctypes.windll.user32.FindWindowW(
                    None, window
                )

                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 5)
                    ctypes.windll.user32.SetForegroundWindow(hwnd)

            key = cv2.waitKey(1) & 0xFF

            try:
                if cv2.getWindowProperty(
                    window, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
            except cv2.error:
                break

            # Accept lowercase and uppercase keyboard keys.
            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("t"), ord("T")):
                tracking = not tracking
                previous_pinch = False
                swipe_start_y = None

                print(
                    "Tracking:",
                    "ENABLED" if tracking else "DISABLED",
                    flush=True,
                )

            elif key in (ord("c"), ord("C")):
                controls_enabled = not controls_enabled

                previous_pinch = False
                swipe_start_y = None
                smooth_x = None
                smooth_y = None

                print(
                    "Computer controls:",
                    "ENABLED" if controls_enabled else "DISABLED",
                    flush=True,
                )

            elif key in (ord("s"), ord("S")):
                path = os.path.join(
                    os.path.dirname(__file__),
                    f"snapshot_{int(time.time())}.jpg",
                )

                cv2.imwrite(path, img)
                print(f"[+] Snapshot saved: {path}", flush=True)

    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False

    finally:
        print("[DEBUG] Releasing webcam resources...", flush=True)

        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

    print("[+] Webcam test completed.", flush=True)
    return True


if __name__ == "__main__":
    headless = "--headless" in sys.argv
    raw = "--raw" in sys.argv

    run_webcam_test(
        headless_frames=10 if headless else 0,
        raw_mode=raw,
    )
