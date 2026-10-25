"""
GestureFlow AI - Environment & Dependency Diagnostic Script
Verifies system compatibility, Python version, disk space, and required package installations.
"""

import sys
import os
import platform
import importlib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def check_disk_space():
    """Check storage availability across Windows logical drives."""
    print("\n--- [1] Checking Disk Space ---")
    try:
        import ctypes
        import string
        
        drives = [f"{d}:\\" for d in string.ascii_uppercase if ctypes.windll.kernel32.GetDriveTypeW(f"{d}:\\") in (2, 3, 4)]
        
        for drive in drives:
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(drive),
                ctypes.byref(free_bytes),
                ctypes.byref(total_bytes),
                None
            )
            free_gb = free_bytes.value / (1024 ** 3)
            total_gb = total_bytes.value / (1024 ** 3)
            
            status = "OK" if free_gb >= 2.0 else "WARNING (<2 GB free)"
            print(f"  Drive {drive} -> {free_gb:.2f} GB free / {total_gb:.2f} GB total [{status}]")
    except Exception as e:
        print(f"  Disk check notice: {e}")

def check_python_environment():
    """Verify Python version and architecture."""
    print("\n--- [2] Checking Python Runtime ---")
    py_ver = sys.version_info
    bitness = platform.architecture()[0]
    print(f"  Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro} ({bitness})")
    print(f"  Executable: {sys.executable}")
    
    if py_ver.major == 3 and py_ver.minor >= 10:
        print("  [PASS] Python version is compatible (>= 3.10).")
    else:
        print("  [WARN] Python >= 3.10 is recommended.")

def check_packages():
    """Verify core dependencies and MediaPipe Tasks model."""
    print("\n--- [3] Checking Required Packages ---")
    packages = {
        "cv2": ("opencv-python", "OpenCV for camera capture and image processing"),
        "mediapipe": ("mediapipe", "MediaPipe for hand landmark tracking"),
        "pyautogui": ("pyautogui", "Safe OS automation and mouse controller"),
        "numpy": ("numpy", "Core numerical operations and matrix math"),
        "sklearn": ("scikit-learn", "Machine learning classifiers (Phase 2)"),
        "streamlit": ("streamlit", "Interactive web dashboard (Phase 5)"),
    }
    
    missing = []
    for import_name, (pkg_name, description) in packages.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  [INSTALLED] {pkg_name:<14} (v{version}) - {description}")
        except ImportError:
            print(f"  [MISSING]   {pkg_name:<14} - {description}")
            missing.append(pkg_name)

    print("\n--- [4] Checking MediaPipe Tasks Model ---")
    from src.config import resolve_model_path
    try:
        model_path = resolve_model_path()
        if os.path.isfile(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  [PASS] Hand Landmarker model found ({size_mb:.2f} MB): {model_path}")
        else:
            print(f"  [MISSING] Model not found at: {model_path}")
    except Exception as err:
        print(f"  [FAIL] Model resolution error: {err}")
            
    print("\n--- Diagnostic Summary ---")
    if not missing:
        print("  All dependencies are verified! Ready for real-time webcam test.")
        return True
    else:
        print(f"  Missing {len(missing)} package(s): {', '.join(missing)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(" GestureFlow AI - System Environment Diagnostic")
    print("=" * 60)
    check_disk_space()
    check_python_environment()
    ready = check_packages()
    print("=" * 60)
    sys.exit(0 if ready else 1)
