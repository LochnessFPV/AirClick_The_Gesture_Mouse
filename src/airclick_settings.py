"""Central, thread-safe settings store for AirClick.

Every tunable value in the application is declared in :data:`SCHEMA`. The schema
doubles as validation metadata and as the description the settings UI uses to
build itself, so adding a new option only requires editing this file.

Settings are persisted to ``~/.airclick/settings.json`` (override with the
``AIRCLICK_SETTINGS`` environment variable) and can be changed while the
application is running: readers take an immutable snapshot which is swapped
atomically whenever a value changes.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("airclick.settings")

SETTINGS_VERSION = 1
DEFAULT_PROFILE = "Default"


@dataclass(frozen=True)
class Setting:
    """Declaration of a single configurable value."""

    default: Any
    kind: str  # "bool" | "int" | "float" | "choice"
    label: str
    help: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Tuple[Any, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "default": self.default,
            "kind": self.kind,
            "label": self.label,
            "help": self.help,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": list(self.choices),
        }


SECTION_LABELS: Dict[str, str] = {
    "pointer": "Pointer",
    "clicks": "Clicks",
    "scroll": "Scroll & Media",
    "camera": "Camera",
    "modes": "Modes",
    "safety": "Safety",
    "general": "General",
}


SCHEMA: Dict[str, Dict[str, Setting]] = {
    "pointer": {
        "speed": Setting(
            1.0, "float", "Pointer speed",
            "How far the cursor travels for a given hand movement.",
            minimum=0.2, maximum=3.0, step=0.1,
        ),
        "smoothness": Setting(
            5, "int", "Smoothness",
            "Higher values give a calmer cursor but react a little slower.",
            minimum=1, maximum=10, step=1,
        ),
        "deadzone": Setting(
            6, "int", "Dead zone (pixels)",
            "Hand tremors smaller than this are ignored completely.",
            minimum=0, maximum=40, step=1,
        ),
        "glide": Setting(
            0.05, "float", "Glide time (seconds)",
            "Time the cursor takes to glide to its new position. 0 is instant.",
            minimum=0.0, maximum=0.3, step=0.01,
        ),
    },
    "clicks": {
        "pinch_sensitivity": Setting(
            0.3, "float", "Pinch sensitivity",
            "Smaller values react to finer pinch movements.",
            minimum=0.05, maximum=1.0, step=0.05,
        ),
        "pinch_hold_frames": Setting(
            5, "int", "Pinch hold (frames)",
            "How long a pinch must be held steady before it triggers.",
            minimum=1, maximum=15, step=1,
        ),
        "gesture_stability_frames": Setting(
            4, "int", "Gesture stability (frames)",
            "How many frames a gesture must persist before it is accepted.",
            minimum=1, maximum=12, step=1,
        ),
        "enable_left_click": Setting(True, "bool", "Left click"),
        "enable_right_click": Setting(True, "bool", "Right click"),
        "enable_double_click": Setting(True, "bool", "Double click"),
        "enable_drag": Setting(True, "bool", "Click and drag"),
    },
    "scroll": {
        "enable_scroll": Setting(True, "bool", "Scrolling"),
        "scroll_step": Setting(
            120, "int", "Scroll speed",
            "Amount scrolled per step.",
            minimum=20, maximum=480, step=20,
        ),
        "enable_volume": Setting(True, "bool", "Volume control"),
        "volume_step": Setting(
            2.0, "float", "Volume step (%)",
            minimum=0.5, maximum=20.0, step=0.5,
        ),
        "enable_brightness": Setting(True, "bool", "Brightness control"),
        "brightness_step": Setting(
            2.0, "float", "Brightness step (%)",
            minimum=0.5, maximum=20.0, step=0.5,
        ),
    },
    "camera": {
        "device_index": Setting(
            0, "int", "Camera device",
            "Which webcam to use. 0 is the built-in camera on most laptops.",
            minimum=0, maximum=10, step=1,
        ),
        "width": Setting(
            640, "choice", "Capture width", choices=(320, 640, 800, 1280, 1920),
        ),
        "height": Setting(
            480, "choice", "Capture height", choices=(240, 480, 600, 720, 1080),
        ),
        "fps_cap": Setting(
            30, "int", "Frame rate limit",
            "Lower values use less battery.",
            minimum=5, maximum=60, step=1,
        ),
        "show_preview": Setting(True, "bool", "Show camera preview"),
        "mirror": Setting(True, "bool", "Mirror the camera image"),
    },
    "modes": {
        "glove_mode": Setting(
            False, "bool", "Glove mode",
            "Track a coloured glove with a marker instead of a bare hand.",
        ),
        "dominant_hand": Setting(
            "right", "choice", "Dominant hand", choices=("right", "left"),
        ),
        "detection_confidence": Setting(
            0.5, "float", "Detection confidence",
            "Raise this if the app sees hands that are not there.",
            minimum=0.1, maximum=0.9, step=0.05,
        ),
        "tracking_confidence": Setting(
            0.5, "float", "Tracking confidence",
            minimum=0.1, maximum=0.9, step=0.05,
        ),
        "voice_assistant": Setting(True, "bool", "Voice assistant"),
    },
    "safety": {
        "failsafe_corner": Setting(
            True, "bool", "Corner failsafe",
            "Move the real mouse to a screen corner to abort automation.",
        ),
        "panic_hotkey": Setting(
            "ctrl+alt+q", "choice", "Panic hotkey",
            "Instantly stops gesture control from anywhere.",
            choices=("ctrl+alt+q", "ctrl+alt+p", "ctrl+shift+q", "none"),
        ),
        "auto_pause_seconds": Setting(
            3.0, "float", "Auto-pause after (seconds)",
            "Pause control when no hand has been seen. 0 disables auto-pause.",
            minimum=0.0, maximum=30.0, step=0.5,
        ),
    },
    "general": {
        "log_level": Setting(
            "INFO", "choice", "Log detail",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        ),
        "start_minimised": Setting(False, "bool", "Start minimised"),
        "confirm_file_open": Setting(
            True, "bool", "Confirm before opening files",
            "Ask for confirmation before the voice assistant opens a file.",
        ),
    },
}


def default_values() -> Dict[str, Dict[str, Any]]:
    """Return a fresh nested dict of every default value."""
    return {
        section: {key: setting.default for key, setting in options.items()}
        for section, options in SCHEMA.items()
    }


def schema_as_dict() -> List[Dict[str, Any]]:
    """Serialisable description of the schema, used to build the settings UI."""
    return [
        {
            "id": section,
            "label": SECTION_LABELS.get(section, section.title()),
            "options": [
                dict(id=key, **setting.as_dict())
                for key, setting in options.items()
            ],
        }
        for section, options in SCHEMA.items()
    ]


class SettingsError(ValueError):
    """Raised when a value cannot be stored for the requested setting."""


def _coerce(section: str, key: str, value: Any) -> Any:
    try:
        setting = SCHEMA[section][key]
    except KeyError as exc:
        raise SettingsError(f"Unknown setting '{section}.{key}'") from exc

    if setting.kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    if setting.kind == "choice":
        for choice in setting.choices:
            if choice == value or str(choice) == str(value):
                return choice
        raise SettingsError(
            f"'{value}' is not a valid option for {section}.{key}"
        )

    try:
        number = int(value) if setting.kind == "int" else float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(
            f"{section}.{key} expects a number, got '{value}'"
        ) from exc

    if setting.minimum is not None:
        number = max(number, type(number)(setting.minimum))
    if setting.maximum is not None:
        number = min(number, type(number)(setting.maximum))
    return number


def settings_path() -> Path:
    """Location of the settings file."""
    override = os.environ.get("AIRCLICK_SETTINGS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".airclick" / "settings.json"


class Settings:
    """Thread-safe, observable settings store with profile support."""

    def __init__(self, path: Optional[Path] = None, autoload: bool = True):
        self._path = Path(path) if path is not None else settings_path()
        self._lock = threading.RLock()
        self._profiles: Dict[str, Dict[str, Dict[str, Any]]] = {
            DEFAULT_PROFILE: default_values()
        }
        self._active = DEFAULT_PROFILE
        self._snapshot: Dict[str, Dict[str, Any]] = copy.deepcopy(
            self._profiles[DEFAULT_PROFILE]
        )
        self._listeners: List[Callable[[str, str, Any], None]] = []
        if autoload:
            self.load()

    # ------------------------------------------------------------------ read

    @property
    def path(self) -> Path:
        return self._path

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Current values.

        The returned dict is replaced (never mutated) on every change, so a
        render loop can hold on to it for the duration of a frame without
        locking.
        """
        return self._snapshot

    def get(self, section: str, key: str) -> Any:
        try:
            return self._snapshot[section][key]
        except KeyError as exc:
            raise SettingsError(f"Unknown setting '{section}.{key}'") from exc

    def section(self, name: str) -> Dict[str, Any]:
        try:
            return self._snapshot[name]
        except KeyError as exc:
            raise SettingsError(f"Unknown settings section '{name}'") from exc

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self._snapshot)

    # ----------------------------------------------------------------- write

    def set(self, section: str, key: str, value: Any, save: bool = True) -> Any:
        """Validate and store a single value, notifying listeners."""
        coerced = _coerce(section, key, value)
        with self._lock:
            values = self._profiles[self._active]
            if values.get(section, {}).get(key) == coerced:
                return coerced
            values.setdefault(section, {})[key] = coerced
            self._snapshot = copy.deepcopy(values)
            if save:
                self._save_locked()
        LOGGER.debug("Setting %s.%s changed to %r", section, key, coerced)
        self._notify(section, key, coerced)
        return coerced

    def update(self, values: Dict[str, Dict[str, Any]], save: bool = True) -> None:
        """Apply many values at once; invalid entries are skipped and logged."""
        changed: List[Tuple[str, str, Any]] = []
        with self._lock:
            current = self._profiles[self._active]
            for section, options in (values or {}).items():
                if section not in SCHEMA or not isinstance(options, dict):
                    LOGGER.warning("Ignoring unknown settings section %r", section)
                    continue
                for key, value in options.items():
                    try:
                        coerced = _coerce(section, key, value)
                    except SettingsError as exc:
                        LOGGER.warning("%s", exc)
                        continue
                    if current.setdefault(section, {}).get(key) != coerced:
                        current[section][key] = coerced
                        changed.append((section, key, coerced))
            if changed:
                self._snapshot = copy.deepcopy(current)
                if save:
                    self._save_locked()
        for section, key, value in changed:
            self._notify(section, key, value)

    def reset(self, section: Optional[str] = None, save: bool = True) -> None:
        """Restore defaults for one section, or for everything."""
        defaults = default_values()
        payload = {section: defaults[section]} if section else defaults
        if section and section not in defaults:
            raise SettingsError(f"Unknown settings section '{section}'")
        self.update(payload, save=save)

    # -------------------------------------------------------------- profiles

    def list_profiles(self) -> List[str]:
        with self._lock:
            return sorted(self._profiles)

    @property
    def active_profile(self) -> str:
        return self._active

    def save_profile(self, name: str) -> None:
        """Store the current values under ``name`` and switch to it."""
        clean = (name or "").strip()
        if not clean:
            raise SettingsError("Profile name cannot be empty")
        with self._lock:
            self._profiles[clean] = copy.deepcopy(self._profiles[self._active])
            self._active = clean
            self._save_locked()

    def activate_profile(self, name: str) -> None:
        with self._lock:
            if name not in self._profiles:
                raise SettingsError(f"Unknown profile '{name}'")
            self._active = name
            self._snapshot = copy.deepcopy(self._profiles[name])
            self._save_locked()
            values = self._snapshot
        for section, options in values.items():
            for key, value in options.items():
                self._notify(section, key, value)

    def delete_profile(self, name: str) -> None:
        with self._lock:
            if name == DEFAULT_PROFILE:
                raise SettingsError("The default profile cannot be deleted")
            if name not in self._profiles:
                raise SettingsError(f"Unknown profile '{name}'")
            del self._profiles[name]
            if self._active == name:
                self._active = DEFAULT_PROFILE
                self._snapshot = copy.deepcopy(self._profiles[DEFAULT_PROFILE])
            self._save_locked()

    # ------------------------------------------------------------- listeners

    def subscribe(self, callback: Callable[[str, str, Any], None]) -> Callable[[], None]:
        """Register ``callback(section, key, value)``; returns an unsubscribe."""
        with self._lock:
            self._listeners.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._listeners:
                    self._listeners.remove(callback)

        return unsubscribe

    def _notify(self, section: str, key: str, value: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for callback in listeners:
            try:
                callback(section, key, value)
            except Exception:  # a bad listener must not break the app
                LOGGER.exception("Settings listener failed for %s.%s", section, key)

    # ----------------------------------------------------------- persistence

    def load(self) -> None:
        """Read the settings file, falling back to defaults when unusable."""
        if not self._path.exists():
            LOGGER.info("No settings file yet; using defaults (%s)", self._path)
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOGGER.exception("Could not read %s; using defaults", self._path)
            return

        profiles_raw = raw.get("profiles")
        if not isinstance(profiles_raw, dict) or not profiles_raw:
            profiles_raw = {DEFAULT_PROFILE: raw.get("values", {})}

        profiles: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for name, stored in profiles_raw.items():
            values = default_values()
            if isinstance(stored, dict):
                for section, options in stored.items():
                    if section not in SCHEMA or not isinstance(options, dict):
                        continue
                    for key, value in options.items():
                        try:
                            values[section][key] = _coerce(section, key, value)
                        except SettingsError as exc:
                            LOGGER.warning("Discarding stored value: %s", exc)
            profiles[str(name)] = values

        profiles.setdefault(DEFAULT_PROFILE, default_values())
        active = str(raw.get("active_profile") or DEFAULT_PROFILE)
        if active not in profiles:
            active = DEFAULT_PROFILE

        with self._lock:
            self._profiles = profiles
            self._active = active
            self._snapshot = copy.deepcopy(profiles[active])
        LOGGER.info("Loaded settings from %s (profile '%s')", self._path, active)

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        payload = {
            "version": SETTINGS_VERSION,
            "active_profile": self._active,
            "profiles": self._profiles,
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            LOGGER.exception("Could not save settings to %s", self._path)


_INSTANCE: Optional[Settings] = None
_INSTANCE_LOCK = threading.Lock()


def get_settings() -> Settings:
    """Return the process-wide settings instance, creating it on first use."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = Settings()
    return _INSTANCE


def configure_logging(level: Optional[str] = None) -> None:
    """Set up console + rotating file logging for the whole application."""
    from logging.handlers import RotatingFileHandler

    if level is None:
        try:
            level = get_settings().get("general", "log_level")
        except Exception:
            level = "INFO"

    root = logging.getLogger("airclick")
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    if root.handlers:
        for handler in root.handlers:
            handler.setLevel(root.level)
        return

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_dir = settings_path().parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "airclick.log", maxBytes=512_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("File logging unavailable; logging to console only")
