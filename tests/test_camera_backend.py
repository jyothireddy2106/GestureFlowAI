"""
GestureFlow AI - Standalone OpenCV Camera Backend Diagnostic (No MediaPipe)
Tests camera index 0 under DirectShow (CAP_DSHOW) and Media Foundation (CAP_MSMF).
Identifies why cap.read() blocks and verifies Win32 GUI window rendering.
"""

import sys
import os
import time
import threading

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV ('opencv-python') is not installed.", flush=True)
    sys.exit(1)


def test_single_backend(cam_idx: int, backend_id: int, backend_name: str, timeout_sec: float = 3.0) -> bool:
    """Test camera initialization, frame capture with timeout watchdog, and GUI rendering."""
    print(f"\n{'='*20} Testing Backend: {backend_name} (Index {cam_idx}) {'='*20}", flush=True)
    
    cap = None
    try:
        # Step 1: VideoCapture creation
        print(f"[DEBUG 1] Before cv2.VideoCapture({cam_idx}, {backend_name})...", flush=True)
        t0 = time.time()
        cap = cv2.VideoCapture(cam_idx, backend_id)
        open_duration_ms = (time.time() - t0) * 1000
        print(f"[DEBUG 1] After cv2.VideoCapture -> took {open_duration_ms:.1f}ms", flush=True)

        # Step 2: isOpened() check
        print("[DEBUG 2] Before cap.isOpened()...", flush=True)
        is_open = cap.isOpened()
        print(f"[DEBUG 2] After cap.isOpened() -> isOpened = {is_open}", flush=True)

        if not is_open:
            print(f"[!] Backend {backend_name} could not open camera index {cam_idx}.", flush=True)
            return False

        # Step 3: Configure resolution BEFORE any read
        print("[DEBUG 3] Applying resolution 640x480...", flush=True)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Optional: Set MJPG FourCC for smooth USB streaming under DSHOW
        if backend_id == cv2.CAP_DSHOW:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        print("[DEBUG 3] Camera properties configured.", flush=True)

        # Step 4: First cap.read() with Thread Watchdog to catch deadlocks
        print(f"[DEBUG 4] Before first cap.read() (watchdog timeout = {timeout_sec}s)...", flush=True)
        read_result = {"success": False, "frame": None, "error": None}

        def read_worker():
            try:
                ret, frame = cap.read()
                read_result["success"] = ret
                read_result["frame"] = frame
            except Exception as ex:
                read_result["error"] = ex

        t_read = time.time()
        th = threading.Thread(target=read_worker, daemon=True)
        th.start()
        th.join(timeout=timeout_sec)
        elapsed_read_ms = (time.time() - t_read) * 1000

        if th.is_alive():
            print(f"[FAIL] cap.read() HUNG for > {timeout_sec}s under {backend_name}! (Deadlock detected)", flush=True)
            return False

        if read_result["error"]:
            print(f"[FAIL] cap.read() raised an error: {read_result['error']}", flush=True)
            return False

        if not read_result["success"] or read_result["frame"] is None:
            print(f"[FAIL] cap.read() returned success=False (took {elapsed_read_ms:.1f}ms).", flush=True)
            return False

        first_frame = read_result["frame"]
        print(f"[PASS] First frame captured in {elapsed_read_ms:.1f}ms! Shape: {first_frame.shape}, Dtype: {first_frame.dtype}", flush=True)

        # Step 5: Test consecutive reads (ensuring stream flow continuity)
        print("[DEBUG 5] Testing 5 consecutive frame reads...", flush=True)
        for i in range(5):
            t_sub = time.time()
            ret, sub_frame = cap.read()
            dt_sub = (time.time() - t_sub) * 1000
            if not ret or sub_frame is None:
                print(f"[FAIL] Subsequent read {i+1} failed!", flush=True)
                return False
            print(f"  Frame {i+1}: captured in {dt_sub:5.1f}ms (shape: {sub_frame.shape})", flush=True)

        # Step 6: Test Win32 GUI window rendering
        print("[DEBUG 6] Testing cv2.namedWindow & cv2.imshow...", flush=True)
        window_name = f"GestureFlow AI - Camera Backend Test ({backend_name})"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # Add visual confirmation banner
        cv2.putText(first_frame, f"Backend: {backend_name} [OK]", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, first_frame)
        print("[DEBUG 6] After cv2.imshow -> Window enqueued to Win32 message queue.", flush=True)

        # Step 7: Test cv2.waitKey(1) Win32 event pump
        print("[DEBUG 7] Before cv2.waitKey(1000) (pumping Win32 messages for 1 second)...", flush=True)
        key = cv2.waitKey(1000) & 0xFF
        print(f"[DEBUG 7] After cv2.waitKey(1000) -> Win32 event pump completed (key={key}).", flush=True)

        print(f"[SUCCESS] Backend {backend_name} verified completely! Frames capture and window renders normally.", flush=True)
        return True

    except Exception as e:
        print(f"[ERROR] Exception during {backend_name} test: {e}", flush=True)
        return False
    finally:
        # Step 8: Strict cleanup
        print("[DEBUG 8] Cleaning up camera and windows...", flush=True)
        if cap is not None:
            try:
                cap.release()
                print("[DEBUG 8.1] cap.release() completed.", flush=True)
            except Exception as ex:
                print(f"[DEBUG 8.1] cap.release() notice: {ex}", flush=True)
        try:
            cv2.destroyAllWindows()
            print("[DEBUG 8.2] cv2.destroyAllWindows() completed.", flush=True)
        except Exception as ex:
            print(f"[DEBUG 8.2] cv2.destroyAllWindows() notice: {ex}", flush=True)


def run_standalone_camera_test():
    print("=" * 65, flush=True)
    print(" GestureFlow AI - Standalone OpenCV Camera Backend Diagnostic", flush=True)
    print("=" * 65, flush=True)
    
    backends = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
    ]

    working_backend = None

    for name, backend_id in backends:
        success = test_single_backend(cam_idx=0, backend_id=backend_id, backend_name=name)
        if success:
            working_backend = (name, backend_id)
            print(f"\n[+] Recommended Backend Confirmed: {name}")
            break
        else:
            print(f"\n[-] Backend {name} failed or timed out.")

    print("\n" + "=" * 65, flush=True)
    print(" Diagnostic Summary", flush=True)
    print("=" * 65, flush=True)
    if working_backend:
        print(f"  Working Backend : {working_backend[0]}", flush=True)
        print("  Next Step       : Integrate this working backend into src/hand_detector & tests/test_webcam.py", flush=True)
    else:
        print("  Working Backend : None (Check camera permissions / hardware connection)", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_standalone_camera_test()
