"""The installed third-party libraries must still provide the APIs we call.

The rest of the suite runs against stubs so it works without a webcam. These
checks deliberately use the real packages, so a breaking upstream release is
caught here instead of at runtime.
"""

import importlib
import sys

import pytest


def _real(name):
    """Return the genuinely installed package, bypassing any conftest stub."""
    existing = sys.modules.get(name)
    if existing is not None and not getattr(existing, "__airclick_stub__", False):
        return existing

    stubs = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == name or key.startswith(name + ".") or name.startswith(key + ".")
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        pytest.skip("{} is not installed".format(name))
    finally:
        for key, module in stubs.items():
            sys.modules.setdefault(key, module)


def test_mediapipe_still_exposes_the_hands_solution():
    """mediapipe 0.10.22+ dropped the solutions API the engine is built on."""
    mp = _real("mediapipe")
    assert hasattr(mp, "solutions"), (
        "mediapipe {} has no solutions API; pin mediapipe<0.10.22".format(
            getattr(mp, "__version__", "unknown")
        )
    )
    assert hasattr(mp.solutions, "hands")
    assert hasattr(mp.solutions, "drawing_utils")


def test_opencv_provides_an_aruco_api_the_glove_engine_can_use():
    cv2 = _real("cv2")
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        pytest.skip("opencv-contrib-python is not installed")
    assert hasattr(aruco, "getPredefinedDictionary") or hasattr(aruco, "Dictionary_get")
    assert hasattr(aruco, "DetectorParameters") or hasattr(aruco, "DetectorParameters_create")
    assert hasattr(aruco, "ArucoDetector") or hasattr(aruco, "detectMarkers")


def test_pyautogui_still_has_the_failsafe_exception():
    pyautogui = _real("pyautogui")
    assert hasattr(pyautogui, "FailSafeException")
    assert hasattr(pyautogui, "PAUSE")


def test_pynput_still_provides_global_hotkeys():
    keyboard = _real("pynput.keyboard")
    assert hasattr(keyboard, "GlobalHotKeys")


@pytest.mark.skipif(sys.platform != "win32", reason="pycaw is Windows only")
def test_pycaw_speaker_device_matches_one_of_the_supported_shapes():
    """pycaw 2023+ returns a wrapper with EndpointVolume; older returns an
    IMMDevice with Activate. The volume adapter handles both, and breaks if a
    release ever offers neither."""
    pycaw = _real("pycaw.pycaw")
    device = pycaw.AudioUtilities.GetSpeakers()
    assert hasattr(device, "EndpointVolume") or hasattr(device, "Activate")
