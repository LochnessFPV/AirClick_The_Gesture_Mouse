"""Glove tracking gesture engine.

An alternative to the bare-hand engine: an ArUco marker on the wrist locates a
coloured glove, and the glove's silhouette is turned into finger counts.

Shares the settings store, logging and lifecycle contract of
:mod:`Gesture_Controller`, so the two engines are interchangeable.
"""

import logging
import math
import os
import glob
import threading
import time

import numpy as np
import cv2
import cv2.aruco as aruco
import pyautogui

from Gesture_Controller import CameraError, open_camera
from airclick_settings import get_settings

LOGGER = logging.getLogger("airclick.glove")

pyautogui.PAUSE = 0

# Camera intrinsics used when no checkerboard calibration images are present.
_FALLBACK_MTX = np.array(
    [[534.34144579, 0.0, 339.15527836],
     [0.0, 534.68425882, 233.84359493],
     [0.0, 0.0, 1.0]]
)
_FALLBACK_DIST = np.array(
    [[-2.88320983e-01, 5.41079685e-02, 1.73501622e-03, -2.61333895e-04, 2.04110465e-01]]
)


def _build_aruco(dict_type, thresh_constant):
    """Create an ArUco dictionary and detector parameters.

    OpenCV 4.7 renamed ``Dictionary_get``/``DetectorParameters_create`` to
    ``getPredefinedDictionary``/``DetectorParameters``; support both so the app
    works on old and current OpenCV releases.
    """
    if hasattr(aruco, "getPredefinedDictionary"):
        aruco_dict = aruco.getPredefinedDictionary(dict_type)
    else:  # OpenCV < 4.7
        aruco_dict = aruco.Dictionary_get(dict_type)

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:  # OpenCV < 4.7
        parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = thresh_constant
    return aruco_dict, parameters


def _detect_markers(gray_frame, aruco_dict, parameters):
    """Detect markers through whichever API this OpenCV build provides."""
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        return detector.detectMarkers(gray_frame)
    return aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)


class Marker:
    def __init__(self, dict_type = aruco.DICT_4X4_50, thresh_constant = 1):
        self.aruco_dict, self.parameters = _build_aruco(dict_type, thresh_constant)
        self.corners = None # corners of Marker
        self.marker_x2y = 1 # width:height ratio
        self.mtx, self.dist = Marker.calibrate()
    
    def calibrate():
        """Estimate camera intrinsics from the bundled checkerboard images.

        Falls back to known-good defaults when the images are missing or no
        board can be found, instead of raising at import time.
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        path = os.path.dirname(os.path.abspath(__file__))
        p1 = os.path.join(path, 'calib_images', 'checkerboard', '*.jpg')
        images = glob.glob(p1)
        gray = None
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                LOGGER.warning("Skipping unreadable calibration image %s", fname)
                continue
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

        if not objpoints or gray is None:
            LOGGER.warning(
                "No usable calibration images in %s; using default camera intrinsics", p1
            )
            return _FALLBACK_MTX, _FALLBACK_DIST

        try:
            _, mtx, dist, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
        except cv2.error:
            LOGGER.warning("Camera calibration failed; using defaults", exc_info=True)
            return _FALLBACK_MTX, _FALLBACK_DIST
        return mtx, dist
    
    def detect(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corners, ids, _ = _detect_markers(gray_frame, self.aruco_dict, self.parameters)
        if ids is None or len(ids) == 0:
            self.corners = None
    
    def is_detected(self):
        return self.corners is not None and len(self.corners) > 0
    
    def draw_marker(self, frame):
        aruco.drawDetectedMarkers(frame, self.corners)
    
        

def ecu_dis(p1, p2):
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

def find_HSV(samples):
    try:
        color = np.uint8([ samples ])
    except (TypeError, ValueError):
        LOGGER.debug("Unusable colour sample, falling back to a default")
        color = np.uint8([ [[105,105,50]] ])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_RGB2HSV)
    return hsv_color

def draw_box(frame, points, color=(0,255,127)):
    if points:
        frame = cv2.line(frame, points[0], points[1], color, thickness=2, lineType=8) #top
        frame = cv2.line(frame, points[1], points[2], color, thickness=2, lineType=8) #right
        frame = cv2.line(frame, points[2], points[3], color, thickness=2, lineType=8) #bottom
        frame = cv2.line(frame, points[3], points[0], color, thickness=2, lineType=8) #left

def in_cam(val, type_):
    if type_ == 'x':
        if val<0:
            return 0
        if val>GestureController.cam_width:
            return GestureController.cam_width
    elif type_ == 'y':
        if val<0:
            return 0
        if val>GestureController.cam_height:
            return GestureController.cam_height
    return val

    
class ROI:
    def __init__(self, roi_alpha1=1.5, roi_alpha2=1.5, roi_beta=2.5, hsv_alpha = 0.3, hsv_beta = 0.5, hsv_lift_up = 0.3):
        self.roi_alpha1 = roi_alpha1
        self.roi_alpha2 = roi_alpha2
        self.roi_beta = roi_beta
        self.roi_corners = None
        
        self.hsv_alpha = hsv_alpha
        self.hsv_beta = hsv_beta
        self.hsv_lift_up = hsv_lift_up
        self.hsv_corners = None
        
        self.marker_top = None
        self.hsv_glove = None
        
    def findROI(self, frame, marker):
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]),int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]),int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]),int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]),int(rec_coor[3][1]))
        
        try:
            marker.marker_x2y = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) / np.sqrt((c3[0]-c2[0])**2 + (c3[1]-c2[1])**2)
        except:
            marker.marker_x2y = 999.0
        
        #mid-point of top line of Marker
        cx = (c1[0] + c2[0])/2
        cy = (c1[1] + c2[1])/2
        
        self.marker_top = [cx, cy]
        
        l = np.absolute(ecu_dis(c1,c4))
        
        try:
            slope_12 = (c1[1]-c2[1])/(c1[0]-c2[0])
        except:
            slope_12 = (c1[1]-c2[1])*999.0 + 0.1
        
        try:
            slope_14 = -1 / slope_12
        except:
            slope_14 = -999.0
        
        if slope_14 < 0:
            sign = 1
        else:
            sign = -1
        
        bot_rx = int(cx + self.roi_alpha2 * l * np.sqrt(1/(1+slope_12**2)))
        bot_ry = int(cy + self.roi_alpha2 * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        bot_lx = int(cx - self.roi_alpha1 * l * np.sqrt(1/(1+slope_12**2)))
        bot_ly = int(cy - self.roi_alpha1 * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        top_lx = int(bot_lx + sign * self.roi_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ly = int(bot_ly + sign * self.roi_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        top_rx = int(bot_rx + sign * self.roi_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ry = int(bot_ry + sign * self.roi_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        bot_lx = in_cam(bot_lx, 'x')
        bot_ly = in_cam(bot_ly, 'y')
        
        bot_rx = in_cam(bot_rx, 'x')
        bot_ry = in_cam(bot_ry, 'y')
        
        top_lx = in_cam(top_lx, 'x')
        top_ly = in_cam(top_ly, 'y')
        
        top_rx = in_cam(top_rx, 'x')
        top_ry = in_cam(top_ry, 'y')
        
        self.roi_corners = [(bot_lx,bot_ly), (bot_rx,bot_ry), (top_rx,top_ry), (top_lx,top_ly)]
        
        
    def find_glove_hsv(self, frame, marker):
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]),int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]),int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]),int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]),int(rec_coor[3][1]))
        
        l = np.absolute(ecu_dis(c1,c4))
        
        try:
            slope_12 = (c1[1]-c2[1])/(c1[0]-c2[0])
        except:
            slope_12 = (c1[1]-c2[1])*999.0 + 0.1
        try:
            slope_14 = -1 / slope_12
        except:
            slope_14 = -999.0
        
        if slope_14 < 0:
            sign = 1
        else:
            sign = -1
               
        bot_rx = int(self.marker_top[0] + self.hsv_alpha * l * np.sqrt(1/(1+slope_12**2)))
        bot_ry = int(self.marker_top[1] - self.hsv_lift_up*l + self.hsv_alpha * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        bot_lx = int(self.marker_top[0] - self.hsv_alpha * l * np.sqrt(1/(1+slope_12**2)))
        bot_ly = int(self.marker_top[1] - self.hsv_lift_up*l - self.hsv_alpha * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        top_lx = int(bot_lx + sign * self.hsv_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ly = int(bot_ly + sign * self.hsv_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        top_rx = int(bot_rx + sign * self.hsv_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ry = int(bot_ry + sign * self.hsv_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        region = frame[top_ry:bot_ry , top_lx:bot_rx]
        b, g, r = np.mean(region, axis=(0, 1))
        
        self.hsv_glove = find_HSV([[r,g,b]])
        self.hsv_corners =  [(bot_lx,bot_ly), (bot_rx,bot_ry), (top_rx,top_ry), (top_lx,top_ly)]
        
    
    def cropROI(self, frame):
        pts = np.array(self.roi_corners)
        
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = frame[y:y+h, x:x+w].copy()
        
        ## (2) make mask
        pts = pts - pts.min(axis=0)
        
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        
        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        
        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        
        kernelOpen = np.ones((3,3),np.uint8)
        kernelClose = np.ones((5,5),np.uint8)
        
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        
        lower_range = np.array([self.hsv_glove[0][0][0]//1-5,50,50])
        upper_range = np.array([self.hsv_glove[0][0][0]//1+5,255,255])
        
        mask = cv2.inRange(hsv, lower_range, upper_range)
        #mask = cv2.dilate(mask,kernelOpen,iterations = 1)
        Opening =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        Closing =cv2.morphologyEx(Opening,cv2.MORPH_CLOSE,kernelClose)
        FinalMask = Closing
        
        return FinalMask


class Glove:
    
    def __init__(self):
        self.fingers = 0
        self.arearatio = 0
        self.gesture = 0
    
    def find_fingers(self, FinalMask):
        conts,h=cv2.findContours(FinalMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        approx, defects = None, None
        if not conts:
            LOGGER.debug("No contours found in the glove mask")
            self.fingers = 0
            return

        try:
            cnt = max(conts, key = lambda x: cv2.contourArea(x))
            #approx the contour a little
            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,epsilon,True)
            #make convex hull around hand
            hull = cv2.convexHull(cnt)
            #define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            if areacnt <= 0:
                self.fingers = 0
                return
            #find the percentage of area not covered by hand in convex hull
            self.arearatio=((areahull-areacnt)/areacnt)*100
            #find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
        except cv2.error:
            LOGGER.debug("Could not analyse the glove contour", exc_info=True)
            self.fingers = 0
            return

        if defects is None or approx is None:
            LOGGER.debug("No convexity defects found in the glove mask")
            self.fingers = 0
            return

        # l = no. of defects
        l=0
        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            if a <= 0 or b <= 0 or c <= 0:
                continue
            s = (a+b+c)/2
            ar = math.sqrt(max(0.0, s*(s-a)*(s-b)*(s-c)))

            #distance between point and convex hull
            d=(2*ar)/a

            # apply cosine rule here
            cosine = (b**2 + c**2 - a**2)/(2*b*c)
            angle = math.acos(max(-1.0, min(1.0, cosine))) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1

            #draw lines around hand
            cv2.line(FinalMask,start, end, [255,255,255], 2)

        l+=1
        self.fingers = l
        
    def find_gesture(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.gesture = 0
        if self.fingers==1:
            #cv2.putText(frame, str(int(arearatio)), (10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            if self.arearatio<15:
                cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 0
            elif self.arearatio<25:
                cv2.putText(frame,'2 fingers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 2
            else:
                cv2.putText(frame,'1 finger',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 1
                    
        elif self.fingers==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            self.gesture = 3
        '''
        elif self.fingers==3:
            #cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif self.fingers==4:
            #cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif self.fingers==5:
            #cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
           # cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        '''

class Tracker:
    def __init__(self):
        self.tracker_started = False
        self.tracker = None
        self.start_time = 0.0
        self.now_time = 0.0
        self.tracker_bbox = None
        
    def corners_to_tracker(self, corners):
        csrt_minX = int( min( [corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]] ))
        csrt_maxX = int( max( [corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]] ))
        csrt_minY = int( min( [corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]] ))
        csrt_maxY = int( max( [corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]] ))
        self.tracker_bbox = [csrt_minX, csrt_minY, csrt_maxX-csrt_minX, csrt_maxY-csrt_minY]
        
    def tracker_to_corner(self, final_bbox):
        if self.tracker_bbox == None:
            return None
        final_bbox = [[[1,2],[3,4],[5,6],[7,8]]]
        final_bbox[0][0] = [self.tracker_bbox[0],self.tracker_bbox[1]]
        final_bbox[0][1] = [self.tracker_bbox[0]+ self.tracker_bbox[2],self.tracker_bbox[1]]
        final_bbox[0][2] = [self.tracker_bbox[0]+ self.tracker_bbox[2],self.tracker_bbox[1] + self.tracker_bbox[3]]
        final_bbox[0][3] = [self.tracker_bbox[0],self.tracker_bbox[1] +self.tracker_bbox[3]]
        return [np.array(final_bbox, dtype = 'f')]
        
    def CSRT_tracker(self, frame):        
        if self.tracker_bbox == None and self.tracker_started == False:
            return
        
        if self.tracker_started == False:
            if self.tracker == None:
                self.tracker = _create_csrt_tracker()
                if self.tracker is None:
                    return
        
        if self.tracker_bbox != None:
            try:
                self.start_time = time.time()
                self.tracker.init(frame, self.tracker_bbox)
                self.tracker_started = True
            except cv2.error:
                LOGGER.debug("Tracker could not be initialised", exc_info=True)
        try:
            ok, self.tracker_bbox = self.tracker.update(frame)
        except cv2.error:
            ok = None
            LOGGER.debug("Tracker update failed", exc_info=True)
        self.now_time = time.time()
        
        if self.now_time-self.start_time >= 2.0 :
            #cv2.putText(frame, "Please posture your hand correctly", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1)
            cv2.putText(frame,'Posture your hand correctly',(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1, cv2.LINE_AA)
            #print("tracking timeout")
            self.tracker_started = False
            self.tracker_bbox = None
            return
            
        if ok:
            # Tracking success
            p1 = (int(self.tracker_bbox[0]), int(self.tracker_bbox[1]))
            p2 = (int(self.tracker_bbox[0] + self.tracker_bbox[2]), int(self.tracker_bbox[1] + self.tracker_bbox[3]))
            cv2.rectangle(frame, p1, p2, (80, 255, 255), 2, 1)
        else :
            # Tracking failure
            self.tracker_started = False
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            LOGGER.debug("Tracking failure detected")


def _create_csrt_tracker():
    """Create a CSRT tracker across the OpenCV builds that move it around."""
    for factory in (
        getattr(cv2, "TrackerCSRT_create", None),
        getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
    ):
        if factory is not None:
            try:
                return factory()
            except cv2.error:
                continue
    LOGGER.warning("CSRT tracking is unavailable in this OpenCV build")
    return None
        
        

class Mouse:
    def __init__(self):
        self.tx_old = 0
        self.ty_old = 0
        self.trial = True
        self.flag = 0

    def reset(self):
        self.trial = True
        self.flag = 0

    def move_mouse(self, frame, position, gesture, config):
        pointer = config["pointer"]
        clicks = config["clicks"]
        speed = float(pointer["speed"])
        deadzone = float(pointer["deadzone"])
        glide = float(pointer["glide"])

        (sx,sy)=pyautogui.size()
        (camx,camy) = (frame.shape[:2][0],frame.shape[:2][1])
        (mx_old,my_old) = pyautogui.position()

        tx = position[0]
        ty = position[1]
        if self.trial:
            self.trial, self.tx_old, self.ty_old = False, tx, ty

        delta_tx = tx - self.tx_old
        delta_ty = ty - self.ty_old
        self.tx_old,self.ty_old = tx,ty

        if (gesture == 3):
            self.flag = 0
            if math.hypot(delta_tx, delta_ty) <= deadzone / max(1.0, sx / camx):
                return
            mx = mx_old + (delta_tx * sx * speed) / (camx * 2)
            my = my_old + (delta_ty * sy * speed) / (camy * 2)
            mx = max(1, min(sx - 2, mx))
            my = max(1, min(sy - 2, my))
            pyautogui.moveTo(int(mx), int(my), duration = glide)

        elif(gesture == 0):
            if self.flag == 0:
                if clicks["enable_double_click"]:
                    pyautogui.doubleClick()
                self.flag = 1
        elif(gesture == 1):
            LOGGER.debug("One finger open")


class GestureController:
    """Glove tracking engine with the same lifecycle contract as the
    bare-hand :class:`Gesture_Controller.GestureController`."""

    gc_mode = 0
    cap = None
    f_start_time = 0
    f_now_time = 0

    cam_width  = 0
    cam_height = 0

    # Built on first use: calibration reads image files from disk and must not
    # run at import time.
    aru_marker = None
    hand_roi = None
    glove = None
    csrt_track = None
    mouse = None

    _stop_event = threading.Event()
    _build_lock = threading.Lock()

    def __init__(self, settings=None, status_callback=None):
        self.settings = settings or get_settings()
        self.status_callback = status_callback
        self._fps = 0.0

    @classmethod
    def stop(cls):
        """Ask a running engine to shut down. Safe to call from any thread."""
        cls._stop_event.set()
        cls.gc_mode = 0

    @classmethod
    def should_run(cls):
        return not cls._stop_event.is_set()

    @classmethod
    def _build_components(cls):
        with cls._build_lock:
            if cls.aru_marker is None:
                cls.aru_marker = Marker()
                cls.hand_roi = ROI(2.5, 2.5, 6, 0.45, 0.6, 0.4)
                cls.glove = Glove()
                cls.csrt_track = Tracker()
                cls.mouse = Mouse()

    def _emit(self, **status):
        if self.status_callback is None:
            return
        payload = {
            "running": bool(GestureController.gc_mode),
            "paused": False,
            "fps": round(self._fps, 1),
        }
        payload.update(status)
        try:
            self.status_callback(payload)
        except Exception:
            LOGGER.exception("Status callback failed")

    def start(self):
        GestureController._stop_event.clear()
        GestureController.gc_mode = 1

        config = self.settings.snapshot()
        pyautogui.FAILSAFE = bool(config["safety"]["failsafe_corner"])

        GestureController._build_components()
        GestureController.mouse.reset()

        capture = None
        camera_config = None
        failed_reads = 0

        LOGGER.info("Glove gesture control started")
        self._emit(message="Glove gesture control started")

        try:
            while GestureController.should_run():
                frame_started = time.perf_counter()
                config = self.settings.snapshot()
                camera = config["camera"]
                pyautogui.FAILSAFE = bool(config["safety"]["failsafe_corner"])

                wanted_camera = (
                    int(camera["device_index"]),
                    int(camera["width"]),
                    int(camera["height"]),
                )
                if wanted_camera != camera_config:
                    if capture is not None:
                        capture.release()
                    capture = open_camera(*wanted_camera)
                    camera_config = wanted_camera
                    GestureController.cap = capture
                    GestureController.cam_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    GestureController.cam_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    LOGGER.info("Using camera %s at %sx%s", *wanted_camera)

                ret, frame = capture.read()
                if not ret or frame is None:
                    failed_reads += 1
                    if failed_reads > 30:
                        raise CameraError(
                            "The camera stopped returning frames. It may have been "
                            "unplugged or taken over by another application."
                        )
                    time.sleep(0.01)
                    continue
                failed_reads = 0

                if camera["mirror"]:
                    frame = cv2.flip(frame, 1)

                #detect Marker, find ROI, find glove HSV, get FinalMask on glove
                FinalMask = None
                GestureController.aru_marker.detect(frame)
                if GestureController.aru_marker.is_detected():
                    GestureController.csrt_track.corners_to_tracker(GestureController.aru_marker.corners)
                    GestureController.csrt_track.CSRT_tracker(frame)
                else:
                    GestureController.csrt_track.tracker_bbox = None
                    GestureController.csrt_track.CSRT_tracker(frame)
                    GestureController.aru_marker.corners = GestureController.csrt_track.tracker_to_corner(GestureController.aru_marker.corners)

                if GestureController.aru_marker.is_detected():
                    try:
                        GestureController.hand_roi.findROI(frame, GestureController.aru_marker)
                        GestureController.hand_roi.find_glove_hsv(frame, GestureController.aru_marker)
                        FinalMask = GestureController.hand_roi.cropROI(frame)
                        GestureController.glove.find_fingers(FinalMask)
                        GestureController.glove.find_gesture(frame)
                        GestureController.mouse.move_mouse(
                            frame,
                            GestureController.hand_roi.marker_top,
                            GestureController.glove.gesture,
                            config,
                        )
                    except (cv2.error, ValueError, IndexError, ZeroDivisionError):
                        LOGGER.debug("Skipping frame, glove could not be analysed",
                                     exc_info=True)
                        FinalMask = None

                if camera["show_preview"]:
                    if GestureController.aru_marker.is_detected():
                        GestureController.aru_marker.draw_marker(frame)
                        draw_box(frame, GestureController.hand_roi.roi_corners, (255,0,0))
                        draw_box(frame, GestureController.hand_roi.hsv_corners, (0,0,250))
                        if FinalMask is not None:
                            cv2.imshow('AirClick Glove Mask', FinalMask)
                    cv2.imshow('AirClick Preview', frame)
                    if cv2.waitKey(1) & 0xFF in (13, 27, ord('q')):
                        LOGGER.info("Preview window closed by the user")
                        break

                elapsed = time.perf_counter() - frame_started
                budget = 1.0 / max(1, int(camera["fps_cap"]))
                if elapsed < budget:
                    time.sleep(budget - elapsed)
                cycle = time.perf_counter() - frame_started
                self._fps = 1.0 / cycle if cycle > 0 else 0.0
                self._emit(gesture=str(GestureController.glove.gesture))

        except pyautogui.FailSafeException:
            LOGGER.warning("Corner failsafe triggered, stopping glove control")
            self._emit(message="Failsafe triggered, gesture control stopped")
        except CameraError as exc:
            LOGGER.error("%s", exc)
            self._emit(message=str(exc))
        except Exception as exc:  # a failure here must not take the app down
            LOGGER.exception("Glove gesture control stopped unexpectedly")
            self._emit(message="Gesture control stopped: {}".format(exc))
        finally:
            GestureController.gc_mode = 0
            GestureController._stop_event.set()
            if capture is not None:
                capture.release()
            GestureController.cap = None
            if camera_config is not None:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            LOGGER.info("Glove gesture control stopped")
            self._emit(running=False, message="Gesture control stopped")


if __name__ == "__main__":
    from airclick_settings import configure_logging

    configure_logging()
    GestureController().start()