import base64

import pytest

from Gesture_Controller import (
    Controller,
    Gest,
    GestureController,
    HandRecog,
    HLabel,
    OneEuroFilter,
    _gesture_label,
    encode_preview,
    parse_resolution,
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


@pytest.fixture
def steady_clock(monkeypatch):
    """Advance time by one frame per call.

    The pointer filter is time-based, so back-to-back calls in a test would
    otherwise be microseconds apart and barely move at all.
    """
    state = {"now": 1000.0}

    def perf_counter():
        state["now"] += 1 / 30.0
        return state["now"]

    monkeypatch.setattr("Gesture_Controller.time.perf_counter", perf_counter)
    return state


def settle(hand, times=25):
    """Run the filter until it has converged on a stationary hand."""
    position = None
    for _ in range(times):
        position = Controller.get_position(hand)
    return position


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


def test_small_movements_inside_the_dead_zone_are_ignored(configured, steady_clock, fake_mouse):
    configured.set("pointer", "mode", "relative")
    configured.set("pointer", "deadzone", 30)
    Controller.configure(configured.snapshot())

    Controller.get_position(hand_at(0.50, 0.50))
    start = fake_mouse.position()
    moved = Controller.get_position(hand_at(0.501, 0.501))
    assert moved == start


def test_relative_mode_pushes_the_cursor_in_the_direction_of_travel(
    configured, steady_clock, fake_mouse
):
    configured.set("pointer", "mode", "relative")
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "smoothness", 1)
    Controller.configure(configured.snapshot())

    settle(hand_at(0.3, 0.3))
    start = fake_mouse.position()
    moved = Controller.get_position(hand_at(0.9, 0.9))
    assert moved[0] > start[0] and moved[1] > start[1]


def test_absolute_mode_puts_the_cursor_where_the_hand_is(
    configured, steady_clock, fake_mouse
):
    configured.set("pointer", "mode", "absolute")
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "smoothness", 1)
    Controller.configure(configured.snapshot())

    width, height = fake_mouse.size()
    centre = settle(hand_at(0.5, 0.5))
    assert abs(centre[0] - width / 2) < width * 0.05
    assert abs(centre[1] - height / 2) < height * 0.05

    Controller.reset_tracking()
    left = settle(hand_at(0.3, 0.5))
    Controller.reset_tracking()
    right = settle(hand_at(0.7, 0.5))
    assert right[0] > left[0]


def test_absolute_mode_can_reach_the_screen_edges(configured, steady_clock, fake_mouse):
    """The hand must not have to leave the camera's view to reach an edge."""
    configured.set("pointer", "mode", "absolute")
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "smoothness", 1)
    Controller.configure(configured.snapshot())

    width, height = fake_mouse.size()
    Controller.reset_tracking()
    assert settle(hand_at(0.1, 0.1)) == (1, 1)
    Controller.reset_tracking()
    assert settle(hand_at(0.9, 0.9)) == (width - 2, height - 2)


def test_higher_speed_needs_less_hand_travel(configured, steady_clock):
    configured.set("pointer", "mode", "absolute")
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "smoothness", 1)

    def travel(speed):
        configured.set("pointer", "speed", speed)
        Controller.configure(configured.snapshot())
        Controller.reset_tracking()
        left = settle(hand_at(0.45, 0.5))
        Controller.reset_tracking()
        right = settle(hand_at(0.55, 0.5))
        return right[0] - left[0]

    assert travel(2.0) > travel(0.5)


def test_the_cursor_never_reaches_a_failsafe_corner(configured, steady_clock, fake_mouse):
    configured.set("pointer", "deadzone", 0)
    configured.set("pointer", "speed", 3.0)
    Controller.configure(configured.snapshot())

    settle(hand_at(0.9, 0.9))
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


# ------------------------------------------------------------ hand routing


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


@pytest.mark.parametrize(
    "text, expected",
    [("640x480", (640, 480)), ("1920x1080", (1920, 1080)), ("nonsense", (640, 480))],
)
def test_resolution_parsing(text, expected):
    assert parse_resolution(text) == expected


# --------------------------------------------------------- pointer filter


def noisy_run(filter_, value, jitter, frames=60):
    """Feed a stationary value with alternating noise and report the spread."""
    outputs = []
    for frame in range(frames):
        sample = value + (jitter if frame % 2 else -jitter)
        outputs.append(filter_(sample, 1000.0 + frame / 30.0))
    tail = outputs[len(outputs) // 2:]
    return max(tail) - min(tail)


def test_the_filter_removes_jitter_from_a_still_hand():
    calm = OneEuroFilter(min_cutoff=0.3)
    assert noisy_run(calm, 500.0, jitter=10.0) < 4.0


def test_more_smoothing_removes_more_jitter():
    snappy = noisy_run(OneEuroFilter(min_cutoff=4.0), 500.0, jitter=10.0)
    calm = noisy_run(OneEuroFilter(min_cutoff=0.3), 500.0, jitter=10.0)
    assert calm < snappy


def test_the_filter_keeps_up_with_deliberate_movement():
    """The whole point: fast movement must not be damped like jitter is."""
    filter_ = OneEuroFilter(min_cutoff=0.3, beta=0.01)
    output = 0.0
    for frame in range(30):
        output = filter_(frame * 60.0, 1000.0 + frame / 30.0)
    target = 29 * 60.0
    assert output > target * 0.8  # tracking, not lagging far behind


def test_the_filter_starts_at_the_first_sample_and_resets():
    filter_ = OneEuroFilter()
    assert filter_(123.0, 1.0) == 123.0
    filter_(456.0, 1.1)
    filter_.reset()
    assert filter_(789.0, 2.0) == 789.0


# ------------------------------------------------------------ hand routing


class FakeResults:
    def __init__(self, labels):
        self.multi_handedness = [
            {"classification": [{"label": label}]} for label in labels
        ]
        self.multi_hand_landmarks = ["hand-" + label for label in labels]


@pytest.fixture
def right_handed(monkeypatch):
    # The fakes stand in for protobuf messages, so skip the real conversion.
    monkeypatch.setattr(
        "Gesture_Controller.MessageToDict", lambda message: message
    )
    previous = GestureController.dom_hand
    GestureController.dom_hand = True
    yield
    GestureController.dom_hand = previous
    GestureController.hr_major = None
    GestureController.hr_minor = None


def test_two_hands_are_split_by_the_dominant_hand_setting(right_handed):
    GestureController.classify_hands(FakeResults(["Right", "Left"]))
    assert GestureController.hr_major == "hand-Right"
    assert GestureController.hr_minor == "hand-Left"


def test_a_lone_non_dominant_hand_still_drives_the_cursor(right_handed):
    """Otherwise a left-handed user gets no control at all: the major hand
    stays None, every gesture reads as the default PALM and nothing happens."""
    GestureController.classify_hands(FakeResults(["Left"]))
    assert GestureController.hr_major == "hand-Left"
    assert GestureController.hr_minor is None


def test_a_lone_dominant_hand_is_unaffected(right_handed):
    GestureController.classify_hands(FakeResults(["Right"]))
    assert GestureController.hr_major == "hand-Right"
    assert GestureController.hr_minor is None


def test_handedness_is_swapped_when_the_image_is_not_mirrored(right_handed):
    """Mediapipe labels hands assuming a selfie view."""
    GestureController.classify_hands(FakeResults(["Right", "Left"]), mirrored=False)
    assert GestureController.hr_major == "hand-Left"
    assert GestureController.hr_minor == "hand-Right"
