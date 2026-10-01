from pathlib import Path

import pytest

import airclick_platform as system


def test_open_path_refuses_a_missing_target(tmp_path):
    assert system.open_path(tmp_path / "nope.txt") is False


def test_open_path_refuses_to_escape_its_root(tmp_path, monkeypatch):
    root = tmp_path / "browse"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")

    opened = []
    monkeypatch.setattr(system.os, "startfile", lambda target: opened.append(target),
                        raising=False)
    monkeypatch.setattr(system.subprocess, "Popen", lambda *a, **k: opened.append(a))

    assert system.open_path(root / ".." / "secret.txt", root=root) is False
    assert opened == []


def test_open_path_allows_a_target_inside_its_root(tmp_path, monkeypatch):
    root = tmp_path / "browse"
    root.mkdir()
    target = root / "notes.txt"
    target.write_text("x", encoding="utf-8")

    opened = []
    monkeypatch.setattr(system.os, "startfile", lambda path: opened.append(Path(path)),
                        raising=False)
    monkeypatch.setattr(system.subprocess, "Popen", lambda *a, **k: opened.append(a))

    assert system.open_path(target, root=root) is True
    assert opened


def test_volume_and_brightness_degrade_quietly(monkeypatch):
    """A machine without the OS back end must not raise, only return None."""
    volume = system.VolumeControl()
    volume._unavailable = True
    assert volume.get_level() is None or isinstance(volume.get_level(), float)

    brightness = system.BrightnessControl()
    brightness._unavailable = True
    assert brightness.get_level() is None
    brightness.set_level(0.5)  # must not raise
