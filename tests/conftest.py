"""Shared test setup.

The application talks to a webcam, the mouse, the speakers and the screen, so
the tests replace those boundaries with deterministic fakes. Heavy optional
dependencies are only stubbed when they are not installed; pyautogui is always
replaced, because a test must never move the real cursor.
"""

import os
import sys
import types

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _module(name, **attributes):
    module = types.ModuleType(name)
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

sys.modules.setdefault("cv2", _module("cv2", CAP_DSHOW=700, CAP_ANY=0))
sys.modules.setdefault(
    "mediapipe",
    _module("mediapipe", solutions=types.SimpleNamespace(drawing_utils=None, hands=None)),
)
sys.modules.setdefault("comtypes", _module("comtypes", CLSCTX_ALL=None))
sys.modules.setdefault("pycaw", _module("pycaw"))
sys.modules.setdefault(
    "pycaw.pycaw", _module("pycaw.pycaw", AudioUtilities=object, IAudioEndpointVolume=object)
)
sys.modules.setdefault("google", _module("google"))
sys.modules.setdefault("google.protobuf", _module("google.protobuf"))
sys.modules.setdefault(
    "google.protobuf.json_format",
    _module("google.protobuf.json_format", MessageToDict=lambda message: message),
)
sys.modules.setdefault("screen_brightness_control", _module("screen_brightness_control"))

sys.modules.setdefault(
    "speech_recognition",
    _module(
        "speech_recognition",
        Recognizer=type("Recognizer", (), {"listen": lambda *a, **k: None}),
        Microphone=object,
        RequestError=type("RequestError", (Exception,), {}),
        UnknownValueError=type("UnknownValueError", (Exception,), {}),
    ),
)
sys.modules.setdefault("pynput", _module("pynput"))
sys.modules.setdefault(
    "pynput.keyboard",
    _module(
        "pynput.keyboard",
        Key=types.SimpleNamespace(ctrl="ctrl"),
        Controller=type("Controller", (), {}),
        GlobalHotKeys=type("GlobalHotKeys", (), {}),
    ),
)

if "eel" not in sys.modules:
    try:
        import eel  # noqa: F401
    except ImportError:
        sys.modules["eel"] = _module(
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
