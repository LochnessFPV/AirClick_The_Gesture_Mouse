"""Hand-tracking gesture engine.

Captures webcam frames, turns MediaPipe hand landmarks into gestures and maps
those gestures onto mouse, scroll, volume and brightness actions.

Everything that used to be a hard-coded constant now comes from
:mod:`airclick_settings` and is re-read every frame, so the settings UI can
retune the engine while it is running.
"""

import base64
import logging
import math
import threading
import time
from enum import IntEnum
from typing import Callable, Dict, List

import cv2
import mediapipe as mp
import pyautogui
from google.protobuf.json_format import MessageToDict

import airclick_platform as system
from airclick_settings import get_settings

LOGGER = logging.getLogger("airclick.gestures")

if not hasattr(mp, "solutions"):
    raise ImportError(
        "This version of mediapipe ({}) has removed the hand solutions API that "
        "AirClick uses. Install a supported release with:\n"
        "    pip install \"mediapipe>=0.10.9,<0.10.22\"".format(
            getattr(mp, "__version__", "unknown")
        )
    )

# Every pyautogui call sleeps for PAUSE seconds by default, adding about 100 ms
# of lag to each cursor update; the frame-rate limiter paces the engine instead.
pyautogui.PAUSE = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Cursor acceleration curve, preserved from the original tuning.
_ACCEL_GAIN = 0.07
_ACCEL_CEILING = 2.1

# In-app preview: small and infrequent enough to stay cheap over the websocket.
_PREVIEW_WIDTH = 320
_PREVIEW_FPS = 12
_PREVIEW_QUALITY = 60

StatusCallback = Callable[[Dict[str, object]], None]


def encode_preview(image):
    """Return a small base64 JPEG of 'image' for display in the app window."""
    height, width = image.shape[:2]
    if width > _PREVIEW_WIDTH:
        scale = _PREVIEW_WIDTH / float(width)
        image = cv2.resize(image, (_PREVIEW_WIDTH, int(height * scale)))
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), _PREVIEW_QUALITY]
    )
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


class OneEuroFilter:
    """Speed-adaptive low-pass filter.

    A fixed amount of smoothing cannot be both steady when the hand is still
    and responsive when it moves: raise it and the cursor lags, lower it and it
    shakes. This filter widens its own cutoff as the hand speeds up, so slow
    movement is heavily damped and fast movement passes through almost
    untouched.

    Casiez, Roussel and Vogel, "1 Euro Filter", CHI 2012.
    """

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._value = None
        self._derivative = 0.0
        self._timestamp = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value, timestamp):
        if self._value is None:
            self._value = value
            self._timestamp = timestamp
            return value

        dt = timestamp - self._timestamp
        if dt <= 0.0:
            dt = 1e-3
        self._timestamp = timestamp

        derivative = (value - self._value) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        self._derivative = alpha_d * derivative + (1.0 - alpha_d) * self._derivative

        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        alpha = self._alpha(cutoff, dt)
        self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value


class CameraError(RuntimeError):
    """Raised when the configured webcam cannot be opened or read."""


# Gesture Encodings 
class Gest(IntEnum):
    # Binary Encoded
    """
    Enum for mapping all hand gesture to binary number.
    """

    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    
    # Extra Mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    """
    Convert Mediapipe Landmarks to recognizable Gestures.
    """
    
    def __init__(self, hand_label, stability_frames: int = 4):
        """
        Constructs all the necessary attributes for the HandRecog object.

        Parameters
        ----------
            finger : int
                Represent gesture corresponding to Enum 'Gest',
                stores computed gesture for current frame.
            ori_gesture : int
                Represent gesture corresponding to Enum 'Gest',
                stores gesture being used.
            prev_gesture : int
                Represent gesture corresponding to Enum 'Gest',
                stores gesture computed for previous frame.
            frame_count : int
                total no. of frames since 'ori_gesture' is updated.
            hand_result : Object
                Landmarks obtained from mediapipe.
            hand_label : int
                Represents multi-handedness corresponding to Enum 'HLabel'.
            stability_frames : int
                Frames a gesture must persist for before it is accepted.
        """

        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
        self.stability_frames = max(1, int(stability_frames))
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        """
        returns signed euclidean distance between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
    
    def get_dist(self, point):
        """
        returns euclidean distance between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    def get_dz(self,point):
        """
        returns absolute difference on z-axis between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    # Function to find Gesture Encoding using current finger_state.
    # Finger_state: 1 if finger is open, else 0
    def set_finger_state(self):
        """
        set 'finger' by computing ratio of distance between finger tip 
        , middle knuckle, base knuckle.

        Returns
        -------
        None
        """
        if self.hand_result is None:
            return

        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0 #thumb
        for point in points:
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])

            # A perfectly straight finger collapses the knuckle-to-knuckle
            # distance to zero; fall back to a small epsilon.
            ratio = round(dist / (dist2 if dist2 else 0.01), 1)

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1
    

    # Handling Fluctations due to noise
    def get_gesture(self):
        """
        returns int representing gesture corresponding to Enum 'Gest'.
        sets 'frame_count', 'ori_gesture', 'prev_gesture', 
        handles fluctations due to noise.
        
        Returns
        -------
        int
        """
        if self.hand_result is None:
            return Gest.PALM

        current_gesture = Gest.PALM
        if self.finger in [Gest.LAST3,Gest.LAST4] and self.get_dist([8,4]) < 0.05:
            if self.hand_label == HLabel.MINOR :
                current_gesture = Gest.PINCH_MINOR
            else:
                current_gesture = Gest.PINCH_MAJOR

        elif Gest.FIRST2 == self.finger :
            dist1 = self.get_dist([8,12])
            dist2 = self.get_dist([5,9])
            ratio = dist1 / dist2 if dist2 else 0.0
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            elif self.get_dz([8,12]) < 0.1:
                current_gesture =  Gest.TWO_FINGER_CLOSED
            else:
                current_gesture =  Gest.MID

        else:
            current_gesture =  self.finger
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count >= self.stability_frames:
            self.ori_gesture = current_gesture
        return self.ori_gesture

# Executes commands according to detected gestures
class Controller:
    """
    Executes commands according to detected gestures.

    All tuning values (pointer speed, smoothing, dead zone, pinch sensitivity,
    step sizes and the per-action on/off switches) are read from the live
    settings snapshot supplied by :class:`GestureController`, so changes made in
    the settings UI take effect on the very next frame.

    Attributes
    ----------
    flag : bool
        true if V gesture is detected
    grabflag : bool
        true if FIST gesture is detected
    pinchmajorflag : bool
        true if PINCH gesture is detected through MAJOR hand,
        on x-axis 'Controller.changesystembrightness', 
        on y-axis 'Controller.changesystemvolume'.
    pinchminorflag : bool
        true if PINCH gesture is detected through MINOR hand,
        on x-axis 'Controller.scrollHorizontal', 
        on y-axis 'Controller.scrollVertical'.
    pinchstartxcoord : int
        x coordinate of hand landmark when pinch gesture is started.
    pinchstartycoord : int
        y coordinate of hand landmark when pinch gesture is started.
    pinchdirectionflag : bool
        true if pinch gesture movment is along x-axis,
        otherwise false
    prevpinchlv : int
        stores quantized magnitued of prev pinch gesture displacment, from 
        starting position
    pinchlv : int
        stores quantized magnitued of pinch gesture displacment, from 
        starting position
    framecount : int
        stores no. of frames since 'pinchlv' is updated.
    prev_hand : tuple
        stores (x, y) coordinates of hand in previous frame.
    smoothed_hand : tuple
        low-pass filtered hand position, used to damp tremor.
    pinch_threshold : float
        step size for quantization of 'pinchlv'.
    """

    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3

    # Pointer smoothing, retuned whenever the Smoothness setting changes.
    _filter_x = OneEuroFilter()
    _filter_y = OneEuroFilter()
    _filter_smoothness = None

    # Settings snapshot for the frame currently being processed.
    _config: Dict[str, Dict[str, object]] = {}

    @classmethod
    def configure(cls, config):
        """Point the controller at the settings snapshot for this frame."""
        cls._config = config
        cls.pinch_threshold = float(config["clicks"]["pinch_sensitivity"])

    @classmethod
    def reset_tracking(cls):
        """Forget pointer history so a returning hand does not fling the cursor."""
        cls.prev_hand = None
        cls._filter_x.reset()
        cls._filter_y.reset()

    @classmethod
    def _tune_filters(cls, smoothness):
        if smoothness == cls._filter_smoothness:
            return
        cls._filter_smoothness = smoothness
        # Smoothness 1 -> 4 Hz cutoff (snappy), 10 -> 0.3 Hz (very calm).
        min_cutoff = 4.0 * (0.075 ** ((smoothness - 1) / 9.0))
        cls._filter_x.min_cutoff = min_cutoff
        cls._filter_y.min_cutoff = min_cutoff

    @classmethod
    def reset(cls):
        """Release any held mouse button and forget per-session state."""
        if cls.grabflag:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                LOGGER.debug("Could not release the mouse button", exc_info=True)
        cls.flag = False
        cls.grabflag = False
        cls.pinchmajorflag = False
        cls.pinchminorflag = False
        cls.framecount = 0
        cls.pinchlv = 0
        cls.prevpinchlv = 0
        cls.reset_tracking()

    def getpinchylv(hand_result):
        """returns distance beween starting pinch y coord and current hand position y coord."""
        dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y)*10,1)
        return dist

    def getpinchxlv(hand_result):
        """returns distance beween starting pinch x coord and current hand position x coord."""
        dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord)*10,1)
        return dist
    
    def changesystembrightness():
        """Nudges screen brightness based on 'Controller.pinchlv'."""
        media = Controller._config.get("scroll", {})
        if not media.get("enable_brightness", True):
            return
        step = float(media.get("brightness_step", 2.0)) / 100.0
        system.brightness.adjust(Controller.pinchlv * step)

    def changesystemvolume():
        """Nudges system volume based on 'Controller.pinchlv'."""
        media = Controller._config.get("scroll", {})
        if not media.get("enable_volume", True):
            return
        step = float(media.get("volume_step", 2.0)) / 100.0
        system.volume.adjust(Controller.pinchlv * step)

    def scrollVertical():
        """scrolls on screen vertically."""
        media = Controller._config.get("scroll", {})
        if not media.get("enable_scroll", True):
            return
        step = int(media.get("scroll_step", 120))
        pyautogui.scroll(step if Controller.pinchlv > 0.0 else -step)

    def scrollHorizontal():
        """scrolls on screen horizontally."""
        media = Controller._config.get("scroll", {})
        if not media.get("enable_scroll", True):
            return
        step = int(media.get("scroll_step", 120))
        pyautogui.keyDown('shift')
        try:
            pyautogui.scroll(-step if Controller.pinchlv > 0.0 else step)
        finally:
            pyautogui.keyUp('shift')

    # Locate Hand to get Cursor Position
    # Stabilize cursor by speed-adaptive filtering and a dead zone
    def get_position(hand_result):
        """
        returns coordinates of current hand position.

        In 'absolute' mode an area of the camera frame maps straight onto the
        screen, like a graphics tablet, so the cursor always corresponds to
        where the hand is. In 'relative' mode the hand nudges the cursor from
        wherever it happens to be, like a mouse, which allows repositioning but
        slowly drifts out of correspondence.

        Returns
        -------
        tuple(int, int)
        """
        pointer = Controller._config.get("pointer", {})
        smoothness = int(pointer.get("smoothness", 5))
        deadzone = float(pointer.get("deadzone", 6))
        speed = max(0.2, float(pointer.get("speed", 1.0)))
        absolute = pointer.get("mode", "absolute") == "absolute"

        Controller._tune_filters(smoothness)

        screen_w, screen_h = pyautogui.size()
        cursor_x, cursor_y = pyautogui.position()
        now = time.perf_counter()

        if absolute:
            # Only the middle of the frame is used, so the hand never has to
            # reach the edge of the camera's view to reach the edge of the
            # screen. A higher speed shrinks the area, needing less hand travel.
            active = max(0.25, min(1.0, 0.7 / speed))
            margin = (1.0 - active) / 2.0
            normal_x = (hand_result.landmark[9].x - margin) / active
            normal_y = (hand_result.landmark[9].y - margin) / active
            x = Controller._filter_x(
                min(1.0, max(0.0, normal_x)) * screen_w, now
            )
            y = Controller._filter_y(
                min(1.0, max(0.0, normal_y)) * screen_h, now
            )
            if math.hypot(x - cursor_x, y - cursor_y) <= deadzone:
                x, y = cursor_x, cursor_y
        else:
            filtered_x = Controller._filter_x(hand_result.landmark[9].x * screen_w, now)
            filtered_y = Controller._filter_y(hand_result.landmark[9].y * screen_h, now)
            if Controller.prev_hand is None:
                Controller.prev_hand = (filtered_x, filtered_y)

            delta_x = filtered_x - Controller.prev_hand[0]
            delta_y = filtered_y - Controller.prev_hand[1]
            Controller.prev_hand = (filtered_x, filtered_y)

            distance = math.hypot(delta_x, delta_y)
            if distance <= deadzone:
                ratio = 0.0
            else:
                ratio = min(_ACCEL_CEILING, _ACCEL_GAIN * distance) * speed
            x = cursor_x + delta_x * ratio
            y = cursor_y + delta_y * ratio

        # Stay a pixel clear of the screen corners so gesture movement never
        # trips the corner failsafe, which is reserved for the physical mouse.
        x = max(1, min(screen_w - 2, x))
        y = max(1, min(screen_h - 2, y))
        return (int(x), int(y))

    def pinch_control_init(hand_result):
        """Initializes attributes for pinch gesture."""
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        Controller.framecount = 0

    # Hold final position for 5 frames to change status
    def pinch_control(hand_result, controlHorizontal, controlVertical):
        """
        calls 'controlHorizontal' or 'controlVertical' based on pinch flags, 
        'framecount' and sets 'pinchlv'.

        Parameters
        ----------
        hand_result : Object
            Landmarks obtained from mediapipe.
        controlHorizontal : callback function assosiated with horizontal
            pinch gesture.
        controlVertical : callback function assosiated with vertical
            pinch gesture. 
        
        Returns
        -------
        None
        """
        if Controller.framecount >= int(Controller._config.get("clicks", {}).get("pinch_hold_frames", 5)):
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv

            if Controller.pinchdirectionflag == True:
                controlHorizontal() #x

            elif Controller.pinchdirectionflag == False:
                controlVertical() #y

        lvx =  Controller.getpinchxlv(hand_result)
        lvy =  Controller.getpinchylv(hand_result)
            
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0

        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    def handle_controls(gesture, hand_result):  
        """Impliments all gesture functionality."""      
        if hand_result is None:
            return

        clicks = Controller._config.get("clicks", {})

        x,y = None,None
        if gesture != Gest.PALM :
            x,y = Controller.get_position(hand_result)
        
        # flag reset
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button = "left")

        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        # implementation
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration = 0)

        elif gesture == Gest.FIST:
            if not clicks.get("enable_drag", True):
                pyautogui.moveTo(x, y, duration = 0)
                return
            if not Controller.grabflag : 
                Controller.grabflag = True
                pyautogui.mouseDown(button = "left")
            pyautogui.moveTo(x, y, duration = 0)

        elif gesture == Gest.MID and Controller.flag:
            if clicks.get("enable_left_click", True):
                pyautogui.click()
            Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            if clicks.get("enable_right_click", True):
                pyautogui.click(button='right')
            Controller.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            if clicks.get("enable_double_click", True):
                pyautogui.doubleClick()
            Controller.flag = False

        elif gesture == Gest.PINCH_MINOR:
            if Controller.pinchminorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result,Controller.scrollHorizontal, Controller.scrollVertical)
        
        elif gesture == Gest.PINCH_MAJOR:
            if Controller.pinchmajorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result,Controller.changesystembrightness, Controller.changesystemvolume)
        
def parse_resolution(value):
    """Turn a '640x480' setting into a (width, height) pair."""
    try:
        width, height = str(value).lower().split("x")
        return int(width), int(height)
    except (AttributeError, TypeError, ValueError):
        LOGGER.warning("Unreadable resolution %r, falling back to 640x480", value)
        return 640, 480


def open_camera(index, width, height):
    """Open a webcam, preferring the fast DirectShow back end on Windows."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if system.IS_WINDOWS else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return capture
        capture.release()
    raise CameraError(
        "Camera {} could not be opened. It may be missing, or in use by another "
        "application.".format(index)
    )


def list_cameras(maximum: int = 5) -> List[int]:
    """Return the indexes of the webcams that can currently be opened."""
    if is_running():
        # Probing would fight the running engine for the device.
        return [int(get_settings().get("camera", "device_index"))]
    available = []
    for index in range(maximum):
        try:
            capture = open_camera(index, 640, 480)
        except CameraError:
            continue
        available.append(index)
        capture.release()
    return available


'''
----------------------------------------  Main Class  ----------------------------------------
    Entry point of Gesture Controller
'''


class GestureController:
    """
    Owns the camera, obtains landmarks from mediapipe and drives 'Controller'.

    Attributes
    ----------
    gc_mode : int
        1 while the engine is running, otherwise 0. Kept for backwards
        compatibility; 'is_running()' is the preferred check.
    cap : Object
        object obtained from cv2, for capturing video frame.
    CAM_HEIGHT : int
        height in pixels of obtained frame from camera.
    CAM_WIDTH : int
        width in pixels of obtained frame from camera.
    hr_major : Object of 'HandRecog'
        object representing major hand.
    hr_minor : Object of 'HandRecog'
        object representing minor hand.
    dom_hand : bool
        True if right hand is dominant hand, otherwise False.
        default True.
    """
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None # Right Hand by default
    hr_minor = None # Left hand by default
    dom_hand = True

    _stop_event = threading.Event()

    def __init__(self, settings=None, status_callback=None, frame_callback=None):
        """Initializes attributes."""
        self.settings = settings or get_settings()
        self.status_callback = status_callback
        self.frame_callback = frame_callback
        self._hotkey_listener = None
        self._paused = False
        self._last_hand_seen = time.time()
        self._last_preview_sent = 0.0
        self._fps = 0.0

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def stop(cls):
        """Ask a running engine to shut down. Safe to call from any thread."""
        cls._stop_event.set()
        cls.gc_mode = 0

    @classmethod
    def should_run(cls):
        return not cls._stop_event.is_set()

    def _emit(self, **status):
        if self.status_callback is None:
            return
        payload = {
            "running": bool(GestureController.gc_mode),
            "paused": self._paused,
            "fps": round(self._fps, 1),
        }
        payload.update(status)
        try:
            self.status_callback(payload)
        except Exception:
            LOGGER.exception("Status callback failed")

    def _send_preview(self, image):
        """Push a frame to the app window, rate limited independently of the loop."""
        if self.frame_callback is None:
            return
        now = time.monotonic()
        if now - self._last_preview_sent < 1.0 / _PREVIEW_FPS:
            return
        self._last_preview_sent = now
        encoded = encode_preview(image)
        if encoded is None:
            return
        try:
            self.frame_callback(encoded)
        except Exception:
            LOGGER.debug("Could not deliver a preview frame", exc_info=True)

    def _start_panic_hotkey(self):
        combo = str(self.settings.get("safety", "panic_hotkey"))
        if combo == "none":
            return
        try:
            from pynput import keyboard

            sequence = "+".join(
                "<{}>".format(part) if part in ("ctrl", "alt", "shift", "cmd") else part
                for part in combo.split("+")
            )
            listener = keyboard.GlobalHotKeys({sequence: self._on_panic})
            listener.daemon = True
            listener.start()
            self._hotkey_listener = listener
            LOGGER.info("Panic hotkey armed: %s", combo)
        except Exception:
            LOGGER.warning("Could not register the panic hotkey %s", combo, exc_info=True)

    def _on_panic(self):
        LOGGER.warning("Panic hotkey pressed, stopping gesture control")
        self._emit(message="Panic hotkey pressed, gesture control stopped")
        GestureController.stop()

    def _stop_panic_hotkey(self):
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                LOGGER.debug("Could not stop the hotkey listener", exc_info=True)
            self._hotkey_listener = None

    # ------------------------------------------------------------ processing

    def classify_hands(results, mirrored=True):
        """
        sets 'hr_major', 'hr_minor' based on classification(left, right) of 
        hand obtained from mediapipe, uses 'dom_hand' to decide major and
        minor hand.

        If only one hand is visible it always becomes the major hand, whichever
        hand it is: requiring the dominant hand would otherwise leave users
        with no control at all when they raise the other one.
        """
        left, right = None, None
        handedness = getattr(results, "multi_handedness", None) or []
        landmarks = getattr(results, "multi_hand_landmarks", None) or []

        for index in range(min(len(handedness), len(landmarks))):
            try:
                label = MessageToDict(handedness[index])["classification"][0]["label"]
            except (KeyError, IndexError, TypeError):
                LOGGER.debug("Unreadable handedness entry %s", index)
                continue
            # Mediapipe labels hands as if the image were a selfie view; with
            # mirroring off the frame is not, so the labels are the wrong way round.
            if not mirrored:
                label = "Left" if label == "Right" else "Right"
            if label == "Right":
                right = landmarks[index]
            else:
                left = landmarks[index]

        if GestureController.dom_hand == True:
            major, minor = right, left
        else :
            major, minor = left, right

        if major is None:
            detected = [hand for hand in (right, left) if hand is not None]
            if len(detected) == 1:
                major, minor = detected[0], None

        GestureController.hr_major = major
        GestureController.hr_minor = minor

    def _apply_runtime_settings(self, config):
        GestureController.dom_hand = config["modes"]["dominant_hand"] == "right"
        pyautogui.FAILSAFE = bool(config["safety"]["failsafe_corner"])
        Controller.configure(config)

    def start(self):
        """
        Capture video frames, obtain landmarks from mediapipe and pass them to
        'handmajor' and 'handminor' for controlling.

        Runs until 'stop()' is called or the preview window is closed, and is
        meant to be executed on its own thread.
        """
        GestureController._stop_event.clear()
        GestureController.gc_mode = 1
        Controller.reset()
        self._paused = False
        self._last_hand_seen = time.time()

        config = self.settings.snapshot()
        self._apply_runtime_settings(config)
        self._start_panic_hotkey()

        capture = None
        hands = None
        hands_config = None
        camera_config = None
        window_open = False
        failed_reads = 0

        stability = int(config["clicks"]["gesture_stability_frames"])
        handmajor = HandRecog(HLabel.MAJOR, stability)
        handminor = HandRecog(HLabel.MINOR, stability)

        LOGGER.info("Gesture control started")
        self._emit(message="Gesture control started")

        try:
            while GestureController.should_run():
                frame_started = time.perf_counter()
                config = self.settings.snapshot()
                self._apply_runtime_settings(config)

                camera = config["camera"]
                width, height = parse_resolution(camera["resolution"])
                wanted_camera = (int(camera["device_index"]), width, height)
                if wanted_camera != camera_config:
                    if capture is not None:
                        capture.release()
                    capture = open_camera(*wanted_camera)
                    camera_config = wanted_camera
                    GestureController.cap = capture
                    GestureController.CAM_WIDTH = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                    GestureController.CAM_HEIGHT = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    LOGGER.info(
                        "Camera %s delivering %gx%g (asked for %sx%s)",
                        wanted_camera[0],
                        GestureController.CAM_WIDTH,
                        GestureController.CAM_HEIGHT,
                        width,
                        height,
                    )

                modes = config["modes"]
                wanted_hands = (
                    float(modes["detection_confidence"]),
                    float(modes["tracking_confidence"]),
                    1 if modes["tracking_quality"] == "accurate" else 0,
                    int(modes["max_hands"]),
                )
                if wanted_hands != hands_config:
                    if hands is not None:
                        hands.close()
                    hands = mp_hands.Hands(
                        max_num_hands=wanted_hands[3],
                        model_complexity=wanted_hands[2],
                        min_detection_confidence=wanted_hands[0],
                        min_tracking_confidence=wanted_hands[1],
                    )
                    hands_config = wanted_hands

                stability = int(config["clicks"]["gesture_stability_frames"])
                handmajor.stability_frames = stability
                handminor.stability_frames = stability

                success, image = capture.read()
                if not success:
                    failed_reads += 1
                    if failed_reads > 30:
                        raise CameraError(
                            "The camera stopped returning frames. It may have been "
                            "unplugged or taken over by another application."
                        )
                    LOGGER.debug("Ignoring empty camera frame")
                    time.sleep(0.01)
                    continue
                failed_reads = 0

                if camera["mirror"]:
                    image = cv2.flip(image, 1)

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)

                gesture_name = "none"
                hands_seen = 0
                if results.multi_hand_landmarks:
                    hands_seen = len(results.multi_hand_landmarks)
                    self._last_hand_seen = time.time()
                    if self._paused:
                        self._paused = False
                        LOGGER.info("Hand detected, resuming")
                        self._emit(message="Resumed")

                    GestureController.classify_hands(results, camera["mirror"])
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)

                    handmajor.set_finger_state()
                    handminor.set_finger_state()
                    gest_name = handminor.get_gesture()

                    if gest_name == Gest.PINCH_MINOR:
                        hand_result = handminor.hand_result
                    else:
                        gest_name = handmajor.get_gesture()
                        hand_result = handmajor.hand_result

                    gesture_name = _gesture_label(gest_name)
                    Controller.handle_controls(gest_name, hand_result)

                    if camera["preview"] != "off":
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )
                else:
                    Controller.reset_tracking()
                    auto_pause = float(config["safety"]["auto_pause_seconds"])
                    if (
                        auto_pause > 0
                        and not self._paused
                        and time.time() - self._last_hand_seen > auto_pause
                    ):
                        self._paused = True
                        Controller.reset()
                        LOGGER.info("No hand seen for %.1fs, pausing", auto_pause)
                        self._emit(message="Paused, no hand detected")

                preview = camera["preview"]
                if preview == "in app window":
                    self._send_preview(image)
                if preview == "separate window":
                    cv2.imshow('AirClick Preview', image)
                    window_open = True
                    if cv2.waitKey(1) & 0xFF in (13, 27):
                        LOGGER.info("Preview window closed by the user")
                        break
                elif window_open:
                    cv2.destroyWindow('AirClick Preview')
                    cv2.waitKey(1)
                    window_open = False

                elapsed = time.perf_counter() - frame_started
                budget = 1.0 / max(1, int(camera["fps_cap"]))
                if elapsed < budget:
                    time.sleep(budget - elapsed)
                cycle = time.perf_counter() - frame_started
                self._fps = 1.0 / cycle if cycle > 0 else 0.0
                self._emit(gesture=gesture_name, hands=hands_seen)

        except pyautogui.FailSafeException:
            LOGGER.warning("Corner failsafe triggered, stopping gesture control")
            self._emit(message="Failsafe triggered, gesture control stopped")
        except CameraError as exc:
            LOGGER.error("%s", exc)
            self._emit(message=str(exc))
        except Exception as exc:  # a failure here must not take the app down
            LOGGER.exception("Gesture control stopped unexpectedly")
            self._emit(message="Gesture control stopped: {}".format(exc))
        finally:
            GestureController.gc_mode = 0
            GestureController._stop_event.set()
            Controller.reset()
            self._stop_panic_hotkey()
            if hands is not None:
                hands.close()
            if capture is not None:
                capture.release()
            GestureController.cap = None
            if window_open:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            LOGGER.info("Gesture control stopped")
            self._emit(running=False, message="Gesture control stopped")


def _gesture_label(gesture):
    """Human readable name for a detected gesture."""
    try:
        return Gest(gesture).name
    except ValueError:
        return str(gesture)


_engine_lock = threading.Lock()
_engine_thread = None


def is_running() -> bool:
    """True while the gesture engine is processing frames."""
    return bool(GestureController.gc_mode)


def start_gesture_control(settings=None, status_callback=None, frame_callback=None) -> bool:
    """Start the engine on a background thread. False if it was already on."""
    global _engine_thread
    with _engine_lock:
        if _engine_thread is not None and _engine_thread.is_alive():
            return False
        controller = GestureController(
            settings=settings,
            status_callback=status_callback,
            frame_callback=frame_callback,
        )
        _engine_thread = threading.Thread(
            target=controller.start, name="airclick-gestures", daemon=True
        )
        _engine_thread.start()
        return True


def stop_gesture_control(timeout: float = 5.0) -> bool:
    """Stop the engine and wait for the camera to be released."""
    global _engine_thread
    with _engine_lock:
        thread = _engine_thread
    if thread is None or not thread.is_alive():
        GestureController.gc_mode = 0
        return False
    GestureController.stop()
    thread.join(timeout)
    if thread.is_alive():
        LOGGER.warning("Gesture engine did not stop within %.1fs", timeout)
    return True


if __name__ == "__main__":
    from airclick_settings import configure_logging

    configure_logging()
    GestureController().start()
