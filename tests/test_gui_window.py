"""
GestureFlow AI - Standalone GUI Window Visibility Diagnostic
Diagnoses OpenCV WIN32UI window rendering, screen positioning, Win32 HWND,
and WND_PROP_VISIBLE state. Keeps window open for 5 seconds with countdown.
"""

import sys
import time
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not installed.", flush=True)
    sys.exit(1)

try:
    import ctypes
    from ctypes import wintypes
    has_ctypes = True
except ImportError:
    has_ctypes = False


def run_gui_diagnostic():
    print("=" * 65, flush=True)
    print(" GestureFlow AI - OpenCV Window Visibility Diagnostic", flush=True)
    print("=" * 65, flush=True)

    # 1. Inspect OpenCV Build Information for GUI
    build_info = cv2.getBuildInformation()
    gui_lines = [line.strip() for line in build_info.splitlines() if "GUI" in line or "Win32 UI" in line]
    print(f"[*] OpenCV Version  : {cv2.__version__}", flush=True)
    print(f"[*] GUI Backend Info: {'; '.join(gui_lines)}", flush=True)

    window_name = "GestureFlow AI - Visibility Diagnostic"

    # 2. Prepare bright, high-visibility test canvas (640x480)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (35, 35, 35) # Dark gray background

    # 3. Create named window explicitly
    print("\n[Step 1] Calling cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)...", flush=True)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    # 4. Explicit screen positioning (forces window to visible coordinate 120, 120)
    print("[Step 2] Positioning window at (120, 120) on screen...", flush=True)
    cv2.moveWindow(window_name, 120, 120)

    # 5. Bring to front and set TOPMOST so it cannot hide behind IDE/terminal
    print("[Step 3] Setting cv2.WND_PROP_TOPMOST = 1...", flush=True)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception as e:
        print(f"[!] setWindowProperty TOPMOST notice: {e}", flush=True)

    # 6. Display frame and pump message loop for 5 seconds
    total_seconds = 5
    start_time = time.time()
    last_sec = -1

    print("\n[Step 4] Starting 5-second display loop with live countdown...", flush=True)

    while True:
        elapsed = time.time() - start_time
        remaining = max(0, total_seconds - int(elapsed))

        if remaining <= 0:
            print("[*] 5-second countdown expired cleanly.", flush=True)
            break

        # Render countdown on canvas
        frame = canvas.copy()
        # Green border
        cv2.rectangle(frame, (10, 10), (630, 470), (46, 204, 113), 3)
        cv2.putText(frame, "GestureFlow AI Window Visibility Test", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Closing in: {remaining} seconds", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
        cv2.putText(frame, "If you see this window, GUI rendering is 100% working!", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (46, 204, 113), 2)
        cv2.putText(frame, "Press any key to close immediately.", (30, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(50) & 0xFF

        # Log state once per second
        current_sec = int(elapsed)
        if current_sec != last_sec:
            last_sec = current_sec

            # Check OpenCV WND_PROP_VISIBLE property
            prop_vis = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            
            # Check Win32 HWND details if ctypes is available
            win32_info = ""
            if has_ctypes:
                hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                if hwnd:
                    # Force window to foreground
                    ctypes.windll.user32.ShowWindow(hwnd, 5) # SW_SHOW
                    ctypes.windll.user32.SetForegroundWindow(hwnd)

                    rect = wintypes.RECT()
                    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    is_visible = ctypes.windll.user32.IsWindowVisible(hwnd)
                    is_iconic = ctypes.windll.user32.IsIconic(hwnd)
                    win32_info = f"HWND={hwnd}, IsWindowVisible={is_visible}, Minimized={is_iconic}, Rect=({rect.left},{rect.top},{rect.right},{rect.bottom})"
                else:
                    win32_info = "HWND not found via FindWindowW"

            print(f"  [T+{current_sec}s] WND_PROP_VISIBLE = {prop_vis} | Win32: {win32_info}", flush=True)

        if key != 255:
            print(f"[*] Key press detected (key={key}). Exiting early.", flush=True)
            break

    cv2.destroyAllWindows()
    print("\n[Step 5] cv2.destroyAllWindows() completed.", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_gui_diagnostic()
