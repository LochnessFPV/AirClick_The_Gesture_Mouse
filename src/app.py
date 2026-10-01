"""Eel bridge between the web UI and the Python back end.

Hosts the chat window, the settings panel and the start/stop controls for the
gesture engine.
"""

import logging
import os
import threading
from queue import Queue

import eel

import Gesture_Controller
from airclick_settings import SettingsError, get_settings, schema_as_dict

LOGGER = logging.getLogger("airclick.app")

_settings = get_settings()

_engine_lock = threading.RLock()
_glove_module = None
_status = {
    "running": False,
    "paused": False,
    "fps": 0.0,
    "gesture": "none",
    "message": "Idle",
}


def _load_glove_module():
    """Import the glove engine on demand.

    It needs ``cv2.aruco``, which only ships with opencv-contrib-python, so a
    missing module must be reported rather than crashing the whole app.
    """
    global _glove_module
    if _glove_module is None:
        import Gesture_Controller_Gloved

        _glove_module = Gesture_Controller_Gloved
    return _glove_module


def _active_engine_module():
    """The module whose engine is currently running, if any."""
    if Gesture_Controller.GestureController.gc_mode:
        return Gesture_Controller
    if _glove_module is not None and _glove_module.GestureController.gc_mode:
        return _glove_module
    return None


def _publish_status(update):
    """Merge an engine status update and push it to the UI."""
    _status.update(update)
    try:
        eel.updateStatus(dict(_status))
    except Exception:
        LOGGER.debug("Could not push status to the UI", exc_info=True)


def start_gesture():
    """Start whichever engine the Glove mode setting selects."""
    with _engine_lock:
        if _active_engine_module() is not None:
            return False, "Gesture control is already running"

        if _settings.get("modes", "glove_mode"):
            try:
                module = _load_glove_module()
            except ImportError:
                LOGGER.exception("Glove mode is unavailable")
                return False, (
                    "Glove mode needs opencv-contrib-python. Install it, or turn "
                    "Glove mode off in Settings."
                )
            controller = module.GestureController(
                settings=_settings, status_callback=_publish_status
            )
            thread = threading.Thread(
                target=controller.start, name="airclick-glove", daemon=True
            )
            thread.start()
            return True, "Glove gesture control started"

        started = Gesture_Controller.start_gesture_control(
            settings=_settings, status_callback=_publish_status
        )
        if not started:
            return False, "Gesture control is already running"
        return True, "Gesture control started"


def stop_gesture():
    """Stop the running engine, whichever one it is."""
    with _engine_lock:
        module = _active_engine_module()
        if module is None:
            return False, "Gesture control is already stopped"
        if module is Gesture_Controller:
            Gesture_Controller.stop_gesture_control()
        else:
            module.GestureController.stop()
        return True, "Gesture control stopped"


def is_gesture_running():
    return _active_engine_module() is not None


class ChatBot:

    started = False
    ready = threading.Event()
    userinputQueue = Queue()

    def isUserInput():
        return not ChatBot.userinputQueue.empty()

    def popUserInput():
        return ChatBot.userinputQueue.get()

    def close_callback(route, websockets):
        if websockets:
            return
        LOGGER.info("UI window closed, shutting down")
        ChatBot.started = False
        ChatBot.ready.set()
        stop_gesture()

    @eel.expose
    def getUserInput(msg):
        ChatBot.userinputQueue.put(msg)
        LOGGER.debug("User input: %s", msg)

    def close():
        ChatBot.started = False
        ChatBot.ready.set()

    def addUserMsg(msg):
        try:
            eel.addUserMsg(msg)
        except Exception:
            LOGGER.debug("Could not deliver a message to the UI", exc_info=True)

    def addAppMsg(msg):
        try:
            eel.addAppMsg(msg)
        except Exception:
            LOGGER.debug("Could not deliver a message to the UI", exc_info=True)

    def start():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
        eel.init(path, allowed_extensions=[".js", ".html"])
        options = dict(
            host="localhost",
            port=27005,
            block=False,
            size=(480, 700),
            position=(10, 100),
            disable_cache=True,
            close_callback=ChatBot.close_callback,
        )
        try:
            try:
                eel.start("index.html", mode="chrome", **options)
            except EnvironmentError:
                # Chrome is not installed; fall back to the default browser.
                LOGGER.info("Chrome not found, opening the UI in the default browser")
                eel.start("index.html", mode="default", **options)

            ChatBot.started = True
            ChatBot.ready.set()
            while ChatBot.started:
                eel.sleep(1.0)
        except Exception:
            LOGGER.exception("The UI could not be started")
        finally:
            ChatBot.started = False
            ChatBot.ready.set()


# ------------------------------------------------------------- UI endpoints


@eel.expose
def getSettingsSchema():
    """Describe every option so the UI can build its own controls."""
    return schema_as_dict()


@eel.expose
def getSettings():
    return {
        "values": _settings.as_dict(),
        "profiles": _settings.list_profiles(),
        "activeProfile": _settings.active_profile,
        "path": str(_settings.path),
    }


@eel.expose
def setSetting(section, key, value):
    try:
        applied = _settings.set(section, key, value)
    except SettingsError as exc:
        LOGGER.warning("Rejected setting %s.%s: %s", section, key, exc)
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "value": applied}


@eel.expose
def resetSettings(section=None):
    try:
        _settings.reset(section)
    except SettingsError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "values": _settings.as_dict()}


@eel.expose
def saveProfile(name):
    try:
        _settings.save_profile(name)
    except SettingsError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "profiles": _settings.list_profiles(),
            "activeProfile": _settings.active_profile}


@eel.expose
def activateProfile(name):
    try:
        _settings.activate_profile(name)
    except SettingsError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "values": _settings.as_dict(),
            "activeProfile": _settings.active_profile}


@eel.expose
def deleteProfile(name):
    try:
        _settings.delete_profile(name)
    except SettingsError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "profiles": _settings.list_profiles(),
            "activeProfile": _settings.active_profile}


@eel.expose
def listCameras():
    try:
        return Gesture_Controller.list_cameras()
    except Exception:
        LOGGER.exception("Could not enumerate cameras")
        return []


@eel.expose
def getStatus():
    _status["running"] = is_gesture_running()
    return dict(_status)


@eel.expose
def startGesture():
    ok, message = start_gesture()
    _publish_status({"message": message, "running": is_gesture_running()})
    return {"ok": ok, "message": message}


@eel.expose
def stopGesture():
    ok, message = stop_gesture()
    _publish_status({"message": message, "running": is_gesture_running()})
    return {"ok": ok, "message": message}
