"""Shared test setup.

The application talks to a webcam, the mouse, the speakers and the screen, so
the tests replace those boundaries with deterministic fakes. Heavy optional
dependencies are only stubbed when they are not installed; pyautogui is always
replaced, because a test must never move the real cursor.
"""

import importlib
import os
import sys
import tempfile
import types

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Redirect the settings file before anything calls get_settings(), so a test
# run never reads or writes the user's real configuration.
os.environ["AIRCLICK_SETTINGS"] = os.path.join(
    tempfile.mkdtemp(prefix="airclick-tests-"), "settings.json"
)


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__airclick_stub__ = True
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


# --- fakes that must always win -------------------------------------------

class FailSafeException(Exception):
    pass


pyautogui_stub = _module("pyautogui")
pyautogui_stub.FAILSAFE = False
pyautogui_stub.PAUSE = 0
pyautogui_stub.FailSafeException = FailSafeException
pyautogui_stub.SCREEN = (1920, 1080)
pyautogui_stub.CURSOR = [960, 540]
pyautogui_stub.calls = []
pyautogui_stub.size = lambda: pyautogui_stub.SCREEN
pyautogui_stub.position = lambda: tuple(pyautogui_stub.CURSOR)


def _record(name):
    def recorder(*args, **kwargs):
        pyautogui_stub.calls.append((name, args, kwargs))

    return recorder


def _move_to(x, y, duration=0):
    pyautogui_stub.CURSOR[:] = [int(x), int(y)]
    pyautogui_stub.calls.append(("moveTo", (int(x), int(y)), {}))


pyautogui_stub.moveTo = _move_to
for _name in ("click", "doubleClick", "mouseDown", "mouseUp", "scroll", "keyDown", "keyUp"):
    setattr(pyautogui_stub, _name, _record(_name))

sys.modules["pyautogui"] = pyautogui_stub


# --- stubs used only when the real package is missing ----------------------


def _ensure(name, **attributes):
    """Import ``name`` for real, falling back to a stub when it is missing.

    Stubbing unconditionally would shadow genuinely installed packages, and a
    fake ``google`` module in particular breaks the real ``google.protobuf``
    namespace package that mediapipe needs.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except Exception:
        sys.modules[name] = _module(name, **attributes)


# pyscreeze (via pyautogui) reads cv2.__version__, so the stub must have one.
_ensure("cv2", CAP_DSHOW=700, CAP_ANY=0, __version__="4.11.0")
if not hasattr(sys.modules["cv2"], "aruco"):
    aruco_stub = _module(
        "cv2.aruco",
        DICT_4X4_50=0,
        getPredefinedDictionary=lambda dict_type: ("dictionary", dict_type),
        DetectorParameters=lambda: types.SimpleNamespace(adaptiveThreshConstant=None),
    )
    sys.modules["cv2.aruco"] = aruco_stub
    sys.modules["cv2"].aruco = aruco_stub
if not hasattr(sys.modules["cv2"], "error"):
    sys.modules["cv2"].error = type("error", (Exception,), {})

_ensure("mediapipe", solutions=types.SimpleNamespace(drawing_utils=None, hands=None))
_ensure("comtypes", CLSCTX_ALL=None)
_ensure("pycaw.pycaw", AudioUtilities=object, IAudioEndpointVolume=object)
_ensure("google.protobuf.json_format", MessageToDict=lambda message: message)
_ensure("screen_brightness_control")
_ensure(
    "speech_recognition",
    Recognizer=type("Recognizer", (), {"listen": lambda *a, **k: None}),
    Microphone=object,
    RequestError=type("RequestError", (Exception,), {}),
    UnknownValueError=type("UnknownValueError", (Exception,), {}),
)
_ensure(
    "pynput.keyboard",
    Key=types.SimpleNamespace(ctrl="ctrl"),
    Controller=type("Controller", (), {}),
    GlobalHotKeys=type("GlobalHotKeys", (), {}),
)
_ensure(
    "eel",
    expose=lambda fn: fn,
    init=lambda *a, **k: None,
    start=lambda *a, **k: None,
    sleep=lambda *a, **k: None,
)


@pytest.fixture(autouse=True)
def reset_cursor():
    pyautogui_stub.CURSOR[:] = [960, 540]
    pyautogui_stub.calls.clear()
    yield
    pyautogui_stub.calls.clear()


@pytest.fixture
def fake_mouse():
    """The pyautogui stub, for asserting on recorded calls."""
    return pyautogui_stub


@pytest.fixture
def settings(tmp_path):
    """A Settings instance backed by a throwaway file."""
    from airclick_settings import Settings

    return Settings(path=tmp_path / "settings.json")
