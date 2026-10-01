"""End-to-end checks of the endpoints the settings UI calls."""

import pytest

app = pytest.importorskip("app")


@pytest.fixture(autouse=True)
def clean_settings():
    app.resetSettings()
    yield
    app.resetSettings()


def test_schema_and_values_line_up():
    schema = app.getSettingsSchema()
    state = app.getSettings()

    for section in schema:
        assert section["id"] in state["values"]
        for option in section["options"]:
            assert option["id"] in state["values"][section["id"]]
            assert option["label"]


def test_setting_a_value_is_applied_and_reported():
    result = app.setSetting("pointer", "smoothness", "8")
    assert result == {"ok": True, "value": 8}
    assert app.getSettings()["values"]["pointer"]["smoothness"] == 8


def test_an_invalid_value_is_rejected_without_raising():
    result = app.setSetting("modes", "dominant_hand", "sideways")
    assert result["ok"] is False
    assert "dominant_hand" in result["error"]


def test_an_unknown_setting_is_rejected():
    assert app.setSetting("nope", "nope", 1)["ok"] is False
    assert app.resetSettings("nope")["ok"] is False


def test_reset_returns_the_defaults():
    app.setSetting("pointer", "speed", 2.5)
    result = app.resetSettings()
    assert result["ok"] is True
    assert result["values"]["pointer"]["speed"] == 1.0


def test_profiles_round_trip():
    app.setSetting("pointer", "speed", 2.0)
    saved = app.saveProfile("Presentation")
    assert saved["ok"] is True
    assert "Presentation" in saved["profiles"]

    app.activateProfile("Default")
    app.setSetting("pointer", "speed", 0.5)

    switched = app.activateProfile("Presentation")
    assert switched["values"]["pointer"]["speed"] == 2.0

    app.activateProfile("Default")
    assert app.deleteProfile("Presentation")["ok"] is True
    assert app.deleteProfile("Default")["ok"] is False


def test_status_reports_a_stopped_engine():
    status = app.getStatus()
    assert status["running"] is False
    assert app.is_gesture_running() is False


def test_stopping_an_idle_engine_is_reported_not_raised():
    result = app.stopGesture()
    assert result["ok"] is False
    assert "stopped" in result["message"]
