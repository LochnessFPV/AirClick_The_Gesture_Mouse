"""The glove engine must import and talk to whichever OpenCV ArUco API is present."""

import types

import pytest

glove = pytest.importorskip("Gesture_Controller_Gloved")


def test_the_module_imports_without_a_camera():
    assert hasattr(glove, "GestureController")
    assert hasattr(glove, "Marker")


def test_modern_aruco_api_is_preferred(monkeypatch):
    calls = {}

    fake = types.SimpleNamespace(
        getPredefinedDictionary=lambda dict_type: calls.setdefault("modern", dict_type),
        DetectorParameters=lambda: types.SimpleNamespace(adaptiveThreshConstant=None),
    )
    monkeypatch.setattr(glove, "aruco", fake)

    _, parameters = glove._build_aruco(7, 3)
    assert calls["modern"] == 7
    assert parameters.adaptiveThreshConstant == 3


def test_legacy_aruco_api_is_used_when_needed(monkeypatch):
    """OpenCV before 4.7 only has the *_get / *_create spellings."""
    calls = {}

    fake = types.SimpleNamespace(
        Dictionary_get=lambda dict_type: calls.setdefault("legacy", dict_type),
        DetectorParameters_create=lambda: types.SimpleNamespace(adaptiveThreshConstant=None),
    )
    monkeypatch.setattr(glove, "aruco", fake)

    _, parameters = glove._build_aruco(5, 1)
    assert calls["legacy"] == 5
    assert parameters.adaptiveThreshConstant == 1


def test_marker_detection_uses_the_detector_object_when_available(monkeypatch):
    class FakeDetector:
        def __init__(self, dictionary, parameters):
            self.dictionary = dictionary

        def detectMarkers(self, frame):
            return ("corners", "ids", "rejected")

    monkeypatch.setattr(glove, "aruco", types.SimpleNamespace(ArucoDetector=FakeDetector))
    assert glove._detect_markers("frame", "dict", "params")[0] == "corners"


def test_marker_detection_falls_back_to_the_free_function(monkeypatch):
    monkeypatch.setattr(
        glove,
        "aruco",
        types.SimpleNamespace(
            detectMarkers=lambda frame, dictionary, parameters: ("old", None, None)
        ),
    )
    assert glove._detect_markers("frame", "dict", "params")[0] == "old"


def test_a_missing_csrt_tracker_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(glove.cv2, "TrackerCSRT_create", None, raising=False)
    monkeypatch.setattr(glove.cv2, "legacy", None, raising=False)
    assert glove._create_csrt_tracker() is None
