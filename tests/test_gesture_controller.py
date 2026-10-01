import base64

import pytest

from Gesture_Controller import (
    Controller,
    Gest,
    HandRecog,
    HLabel,
    _gesture_label,
    encode_preview,
)


class FakeLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z


class FakeHandResult:
    def __init__(self, landmarks):
        self.landmark = landmarks


def hand_at(x, y):
    """A hand result whose landmark 9 (the cursor anchor) sits at x, y."""
    return FakeHandResult([FakeLandmark(x, y) for _ in range(21)])


@pytest.fixture
def configured(settings):
    Controller.configure(settings.snapshot())
    Controller.reset()
    yield settings
    Controller.reset()


# --------------------------------------------------------------- HandRecog


def test_get_signed_dist_positive_and_negative():
    recog = HandRecog(HLabel.MAJOR)
    recog.update_hand_result(FakeHandResult([FakeLandmark(0, 0), FakeLandmark(0, 1)]))
    assert recog.get_signed_dist([0, 1]) > 0

    recog.update_hand_result(FakeHandResult([FakeLandmark(0, 1), FakeLandmark(0, 0)]))
    assert recog.get_signed_dist([0, 1]) < 0


def test_set_finger_state_survives_collapsed_knuckles():
    """A perfectly straight finger makes the reference distance zero."""
    recog = HandRecog(HLabel.MAJOR)
    recog.update_hand_result(FakeHandResult([FakeLandmark(0.5, 0.5) for _ in range(21)]))
    recog.set_finger_state()
    assert recog.finger == 0


def test_gesture_is_only_accepted_once_it_is_stable():
    recog = HandRecog(HLabel.MAJOR, stability_frames=3)
    recog.update_hand_result(hand_at(0.5, 0.5))
    recog.finger = Gest.FIST

    assert recog.get_gesture() == Gest.PALM  # first sighting, not yet trusted
    recog.get_gesture()
    recog.get_gesture()
    assert recog.get_gesture() == Gest.FIST


def test_gesture_label_handles_unmapped_values():
    assert _gesture_label(Gest.FIST) == "FIST"
    assert _gesture_label(3) == "3"


# -------------------------------------------------------------- Controller


def test_small_movements_inside_the_dead_zone_are_ignored(configured, fake_mouse):
    configured.set("pointer", "deadzone", 30)
    configured.set("pointer", "smoothness", 1)
    Controller.configure(configured.snapshot())

    Controller.get_position(hand_at(0.50, 0.50))
    start = fake_mouse.position()
    moved = Controller.get_position(hand_at(0.501, 0.501))
    assert moved == start


def test_larger_movements_move_the_cursor(configured, fake_mouse):
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "smoothness", 1)
    Controller.configure(configured.snapshot())

    Controller.get_position(hand_at(0.3, 0.3))
    start = fake_mouse.position()
    moved = Controller.get_position(hand_at(0.6, 0.6))
    assert moved[0] > start[0] and moved[1] > start[1]


def test_higher_speed_moves_the_cursor_further(configured):
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "smoothness", 1)

    def travel(speed):
        configured.set("pointer", "speed", speed)
        Controller.configure(configured.snapshot())
        Controller.reset()
        Controller.get_position(hand_at(0.4, 0.5))
        before = Controller.get_position(hand_at(0.4, 0.5))
        after = Controller.get_position(hand_at(0.6, 0.5))
        return after[0] - before[0]

    assert travel(2.0) > travel(0.5)


def test_the_cursor_never_reaches_a_failsafe_corner(configured, fake_mouse):
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "speed", 3.0)
    Controller.configure(configured.snapshot())

    Controller.get_position(hand_at(0.9, 0.9))
    for _ in range(40):
        x, y = Controller.get_position(hand_at(0.0, 0.0))
        fake_mouse.CURSOR[:] = [x, y]

    width, height = fake_mouse.size()
    assert 0 < x < width - 1
    assert 0 < y < height - 1


def test_handle_controls_ignores_a_missing_hand(configured, fake_mouse):
    Controller.handle_controls(Gest.V_GEST, None)
    assert fake_mouse.calls == []


def test_disabled_actions_do_nothing(configured, fake_mouse):
    configured.set("clicks", "enable_right_click", False)
    configured.set("scroll", "enable_scroll", False)
    Controller.configure(configured.snapshot())

    Controller.flag = True
    Controller.handle_controls(Gest.INDEX, hand_at(0.5, 0.5))
    Controller.scrollVertical()
    assert fake_mouse.calls == []


def test_enabled_right_click_is_forwarded(configured, fake_mouse):
    Controller.flag = True
    Controller.handle_controls(Gest.INDEX, hand_at(0.5, 0.5))
    assert ("click", (), {"button": "right"}) in fake_mouse.calls


def test_reset_releases_a_held_mouse_button(configured, fake_mouse):
    Controller.grabflag = True
    Controller.reset()
    assert ("mouseUp", (), {"button": "left"}) in fake_mouse.calls
    assert Controller.grabflag is False


def test_scroll_step_follows_the_setting(configured, fake_mouse):
    configured.set("scroll", "scroll_step", 240)
    Controller.configure(configured.snapshot())
    Controller.pinchlv = 1.0
    Controller.scrollVertical()
    assert ("scroll", (240,), {}) in fake_mouse.calls


# ----------------------------------------------------------------- preview


def test_preview_frames_are_shrunk_and_jpeg_encoded():
    numpy = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "imencode"):
        pytest.skip("cv2 is stubbed in this environment")

    frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
    encoded = encode_preview(frame)

    assert isinstance(encoded, str) and encoded
    decoded = cv2.imdecode(
        numpy.frombuffer(base64.b64decode(encoded), dtype=numpy.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape[1] == 320  # downscaled for the websocket
    assert len(encoded) < 40_000


def test_the_preview_setting_offers_all_three_destinations(settings):
    for choice in ("in app window", "separate window", "off"):
        assert settings.set("camera", "preview", choice) == choice
