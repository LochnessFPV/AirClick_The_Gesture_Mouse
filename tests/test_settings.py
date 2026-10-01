import json

import pytest

from airclick_settings import (
    DEFAULT_PROFILE,
    Settings,
    SettingsError,
    default_values,
    schema_as_dict,
)


def test_defaults_match_the_schema(settings):
    assert settings.as_dict() == default_values()


def test_numbers_are_clamped_to_their_range(settings):
    assert settings.set("pointer", "speed", 99) == 3.0
    assert settings.set("pointer", "speed", -5) == 0.2


def test_values_are_coerced_to_the_declared_type(settings):
    assert settings.set("camera", "fps_cap", "24") == 24
    assert settings.set("camera", "resolution", "1280x720") == "1280x720"
    assert settings.set("clicks", "enable_drag", "false") is False
    assert settings.set("clicks", "enable_drag", "yes") is True


def test_unknown_or_invalid_values_are_rejected(settings):
    with pytest.raises(SettingsError):
        settings.set("modes", "dominant_hand", "sideways")
    with pytest.raises(SettingsError):
        settings.set("pointer", "nonexistent", 1)
    with pytest.raises(SettingsError):
        settings.set("nonexistent", "speed", 1)


def test_snapshot_is_replaced_not_mutated(settings):
    before = settings.snapshot()
    settings.set("pointer", "smoothness", 9)
    assert before is not settings.snapshot()
    assert before["pointer"]["smoothness"] == 5


def test_listeners_are_notified_of_changes(settings):
    seen = []
    unsubscribe = settings.subscribe(lambda *change: seen.append(change))

    settings.set("pointer", "deadzone", 12)
    assert seen == [("pointer", "deadzone", 12)]

    settings.set("pointer", "deadzone", 12)  # unchanged, so no notification
    assert len(seen) == 1

    unsubscribe()
    settings.set("pointer", "deadzone", 3)
    assert len(seen) == 1


def test_a_failing_listener_does_not_break_writes(settings):
    settings.subscribe(lambda *change: 1 / 0)
    assert settings.set("pointer", "deadzone", 7) == 7


def test_values_survive_a_restart(tmp_path):
    path = tmp_path / "settings.json"
    first = Settings(path=path)
    first.set("pointer", "speed", 2.5)
    first.set("camera", "fps_cap", 15)

    second = Settings(path=path)
    assert second.get("pointer", "speed") == 2.5
    assert second.get("camera", "fps_cap") == 15


def test_a_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert Settings(path=path).as_dict() == default_values()


def test_out_of_range_stored_values_are_repaired(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"profiles": {DEFAULT_PROFILE: {"pointer": {"speed": 500}}}}),
        encoding="utf-8",
    )
    assert Settings(path=path).get("pointer", "speed") == 3.0


def test_reset_restores_one_section_only(settings):
    settings.set("pointer", "speed", 2.0)
    settings.set("camera", "fps_cap", 10)
    settings.reset("pointer")
    assert settings.get("pointer", "speed") == 1.0
    assert settings.get("camera", "fps_cap") == 10


def test_profiles_can_be_saved_switched_and_deleted(settings):
    settings.set("pointer", "speed", 2.0)
    settings.save_profile("Presentation")
    assert settings.active_profile == "Presentation"

    settings.set("pointer", "speed", 0.5)
    settings.activate_profile(DEFAULT_PROFILE)
    assert settings.get("pointer", "speed") == 2.0

    settings.delete_profile("Presentation")
    assert settings.list_profiles() == [DEFAULT_PROFILE]

    with pytest.raises(SettingsError):
        settings.delete_profile(DEFAULT_PROFILE)


def test_schema_describes_every_default():
    described = {
        (section["id"], option["id"])
        for section in schema_as_dict()
        for option in section["options"]
    }
    actual = {
        (section, key)
        for section, options in default_values().items()
        for key in options
    }
    assert described == actual
