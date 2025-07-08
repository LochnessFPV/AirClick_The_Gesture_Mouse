import os
import sys
import types

# Allow importing from src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Stub heavy optional dependencies required by Gesture_Controller
sys.modules.setdefault('cv2', types.ModuleType('cv2'))

mp_stub = types.ModuleType('mediapipe')
mp_stub.solutions = types.SimpleNamespace(drawing_utils=None, hands=None)
sys.modules.setdefault('mediapipe', mp_stub)

pyautogui_stub = types.ModuleType('pyautogui')
pyautogui_stub.FAILSAFE = False
sys.modules.setdefault('pyautogui', pyautogui_stub)

comtypes_stub = types.ModuleType('comtypes')
comtypes_stub.CLSCTX_ALL = None
sys.modules.setdefault('comtypes', comtypes_stub)

pycaw_pkg = types.ModuleType('pycaw')
pycaw_mod = types.ModuleType('pycaw.pycaw')
pycaw_mod.AudioUtilities = object
pycaw_mod.IAudioEndpointVolume = object
sys.modules.setdefault('pycaw', pycaw_pkg)
sys.modules.setdefault('pycaw.pycaw', pycaw_mod)

google_pkg = types.ModuleType('google')
protobuf_pkg = types.ModuleType('google.protobuf')
json_format_mod = types.ModuleType('google.protobuf.json_format')
json_format_mod.MessageToDict = lambda x: {}
sys.modules.setdefault('google', google_pkg)
sys.modules.setdefault('google.protobuf', protobuf_pkg)
sys.modules.setdefault('google.protobuf.json_format', json_format_mod)

sys.modules.setdefault('screen_brightness_control', types.ModuleType('screen_brightness_control'))

from Gesture_Controller import HandRecog, HLabel

class FakeLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class FakeHandResult:
    def __init__(self, landmarks):
        self.landmark = landmarks


def test_get_signed_dist_positive_and_negative():
    recog = HandRecog(HLabel.MAJOR)
    # first point above second -> positive distance
    landmarks = [FakeLandmark(0, 0), FakeLandmark(0, 1)]
    recog.update_hand_result(FakeHandResult(landmarks))
    dist = recog.get_signed_dist([0, 1])
    assert dist > 0

    # first point below second -> negative distance
    landmarks = [FakeLandmark(0, 1), FakeLandmark(0, 0)]
    recog.update_hand_result(FakeHandResult(landmarks))
    dist = recog.get_signed_dist([0, 1])
    assert dist < 0
