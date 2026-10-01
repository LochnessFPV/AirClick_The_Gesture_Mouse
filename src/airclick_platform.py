"""Operating-system integration for AirClick.

The gesture engine needs to change the system volume, change the screen
brightness, speak and open files. Those are the only parts of the project that
are genuinely platform specific, so they are isolated here: on an unsupported
platform the feature degrades into a logged no-op instead of crashing the app.

Everything that can block (COM calls, brightness ramps, speech) is executed on
a dedicated worker thread so the camera loop never stalls.
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

LOGGER = logging.getLogger("airclick.platform")

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


class _Worker:
    """Serialises blocking system calls onto a single daemon thread."""

    def __init__(self, name: str, max_pending: int = 8):
        self._queue: "queue.Queue[Optional[Callable[[], None]]]" = queue.Queue(max_pending)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self._lock = threading.Lock()

    def submit(self, job: Callable[[], None]) -> bool:
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            # The user is gesturing faster than the OS can keep up; dropping the
            # request keeps the cursor responsive.
            LOGGER.debug("Dropping system request, worker is busy")
            return False

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                job()
            except Exception:
                LOGGER.exception("System call failed")
            finally:
                self._queue.task_done()


_SYSTEM_WORKER = _Worker("airclick-system")


# --------------------------------------------------------------------- volume


class VolumeControl:
    """Reads and writes the master output volume as a 0.0-1.0 scalar."""

    def __init__(self) -> None:
        self._interface = None
        self._unavailable = False

    def _windows_interface(self):
        if self._interface is not None or self._unavailable:
            return self._interface
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._interface = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:
            LOGGER.warning("System volume control is unavailable", exc_info=True)
            self._unavailable = True
        return self._interface

    def get_level(self) -> Optional[float]:
        if IS_WINDOWS:
            interface = self._windows_interface()
            if interface is None:
                return None
            try:
                return float(interface.GetMasterVolumeLevelScalar())
            except Exception:
                LOGGER.warning("Could not read system volume", exc_info=True)
                return None
        if IS_MACOS:
            out = _run(["osascript", "-e", "output volume of (get volume settings)"])
            if out is None:
                return None
            try:
                return max(0.0, min(1.0, float(out.strip()) / 100.0))
            except ValueError:
                return None
        return None

    def set_level(self, level: float) -> None:
        level = max(0.0, min(1.0, float(level)))
        if IS_WINDOWS:
            interface = self._windows_interface()
            if interface is None:
                return
            try:
                interface.SetMasterVolumeLevelScalar(level, None)
            except Exception:
                LOGGER.warning("Could not set system volume", exc_info=True)
            return
        if IS_MACOS:
            _run(["osascript", "-e", f"set volume output volume {int(level * 100)}"])
            return
        if IS_LINUX and shutil.which("amixer"):
            _run(["amixer", "-q", "sset", "Master", f"{int(level * 100)}%"])
            return
        LOGGER.debug("Volume control is not supported on this platform")

    def adjust(self, delta: float) -> None:
        """Change the volume by ``delta`` (0.0-1.0 scale), without blocking."""

        def job() -> None:
            current = self.get_level()
            if current is None:
                if IS_LINUX and shutil.which("amixer"):
                    sign = "+" if delta >= 0 else "-"
                    _run(["amixer", "-q", "sset", "Master", f"{abs(delta) * 100:.0f}%{sign}"])
                return
            self.set_level(current + delta)

        _SYSTEM_WORKER.submit(job)


# ----------------------------------------------------------------- brightness


class BrightnessControl:
    """Reads and writes display brightness as a 0.0-1.0 scalar."""

    def __init__(self) -> None:
        self._module = None
        self._unavailable = False

    def _backend(self):
        if self._module is not None or self._unavailable:
            return self._module
        try:
            import screen_brightness_control as sbc

            self._module = sbc
        except Exception:
            LOGGER.warning("Screen brightness control is unavailable", exc_info=True)
            self._unavailable = True
        return self._module

    def get_level(self) -> Optional[float]:
        sbc = self._backend()
        if sbc is None:
            return None
        try:
            value = sbc.get_brightness(display=0)
            if isinstance(value, (list, tuple)):  # API changed in 0.10
                value = value[0] if value else None
            if value is None:
                return None
            return max(0.0, min(1.0, float(value) / 100.0))
        except Exception:
            LOGGER.warning("Could not read screen brightness", exc_info=True)
            self._unavailable = True
            return None

    def set_level(self, level: float) -> None:
        sbc = self._backend()
        if sbc is None:
            return
        try:
            sbc.set_brightness(int(max(0.0, min(1.0, level)) * 100), display=0)
        except Exception:
            LOGGER.warning("Could not set screen brightness", exc_info=True)
            self._unavailable = True

    def adjust(self, delta: float) -> None:
        """Change brightness by ``delta`` (0.0-1.0 scale), without blocking."""

        def job() -> None:
            current = self.get_level()
            if current is None:
                return
            self.set_level(current + delta)

        _SYSTEM_WORKER.submit(job)


# ------------------------------------------------------------------- speaking


class Speaker:
    """Queued text-to-speech.

    ``pyttsx3`` engines are not re-entrant: calling ``runAndWait()`` from two
    threads, or while it is already speaking, raises or deadlocks. All speech
    therefore goes through one worker thread.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._engine = None
        self._unavailable = False

    def _ensure_engine(self):
        if self._engine is not None or self._unavailable:
            return self._engine
        try:
            import pyttsx3

            driver = "sapi5" if IS_WINDOWS else ("nsss" if IS_MACOS else None)
            try:
                engine = pyttsx3.init(driver) if driver else pyttsx3.init()
            except Exception:
                engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            self._engine = engine
        except Exception:
            LOGGER.warning("Text to speech is unavailable; messages are text only",
                           exc_info=True)
            self._unavailable = True
        return self._engine

    def say(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="airclick-speech", daemon=True
                )
                self._thread.start()
        self._queue.put(text)

    def shutdown(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            engine = self._ensure_engine()
            if engine is None:
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                LOGGER.exception("Speech failed")


# ----------------------------------------------------------------- filesystem


def open_path(target: Path, root: Optional[Path] = None) -> bool:
    """Open ``target`` with the system handler.

    ``root`` restricts what may be opened: the resolved target must stay inside
    it, which prevents a mis-heard voice command from escaping the browsed
    folder via ``..`` segments.
    """
    try:
        resolved = Path(target).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        LOGGER.warning("Refusing to open missing path %s", target)
        return False

    if root is not None:
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
        except ValueError:
            LOGGER.warning("Refusing to open %s: outside of %s", resolved, root)
            return False

    try:
        if IS_WINDOWS:
            os.startfile(str(resolved))  # noqa: S606 - resolved, validated path
        elif IS_MACOS:
            subprocess.Popen(["open", str(resolved)])
        else:
            subprocess.Popen(["xdg-open", str(resolved)])
        return True
    except Exception:
        LOGGER.exception("Could not open %s", resolved)
        return False


def _run(command) -> Optional[str]:
    """Run a command without a shell and return its stdout, or None."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.debug("Command failed: %s", command, exc_info=True)
        return None
    if result.returncode != 0:
        LOGGER.debug("Command %s exited with %s", command[0], result.returncode)
        return None
    return result.stdout


volume = VolumeControl()
brightness = BrightnessControl()
speaker = Speaker()
