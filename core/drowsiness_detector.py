from __future__ import annotations
import time
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

from utils.config import get_config
from core.posture_analyzer import PostureAnalyzer


class DrowsinessDetector:
    def __init__(self, config_path: str = "config/settings.json"):
        cfg = get_config(config_path)

        # ---- Ngưỡng ----
        self.ear_th = float(cfg["DROWSY_THRESHOLD"])
        self.ear_duration_th = float(cfg.get("drowsy_ear_duration", 2.0))
        self.max_head_pitch = float(cfg["max_head_updown_angle"])
        self.max_head_yaw = float(cfg["max_head_side_angle"])
        self.max_shoulder_tilt = float(cfg["max_shoulder_tilt"])
        self.min_gaze_distance_cm = float(cfg["MIN_REASONABLE_DISTANCE"])
        self.max_gaze_distance_cm = float(cfg["MAX_REASONABLE_DISTANCE"])
        self.posture_window_sec = 3.0
        self.gaze_off_threshold_sec = 2.0

        # ---- EAR debounce ----
        self._ear_buf = deque(maxlen=5)
        self._ear_low_start: Optional[float] = None
        self._ear_low_frames: int = 0
        self.EAR_CONSEC_FRAMES: int = 3  # ~100 ms ở 30 FPS

        # ---- Posture & Gaze ----
        self._posture_bad_start: Optional[float] = None
        self._gaze_off_start: Optional[float] = None
        self._gaze_last_seen: Optional[float] = None
        self.MAX_MISSING_DIST_SEC: float = 1.0

        # ---- Drowsiness state + hysteresis ----
        self._drowsy: bool = False
        self._drowsy_end_time: Optional[float] = None
        self.DROWSY_RELEASE_SEC: float = 1.0  # giữ trạng thái thêm 1 s

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def update(
        self,
        ear: Optional[float] = None,
        posture_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        info: Dict[str, Any] = {
            "drowsiness_detected": False,
            "reason": None,
            "ear_duration": 0.0,
            "posture_bad_duration": 0.0,
            "gaze_off_duration": 0.0,
        }

        # 1) EAR debounce
        ear_drowsy = False
        if ear is not None:
            ear_f = self._filter_ear(ear, self._ear_buf)
            ear_low_now = ear_f < self.ear_th

            if ear_low_now:
                self._ear_low_frames += 1
                if self._ear_low_start is None and self._ear_low_frames >= self.EAR_CONSEC_FRAMES:
                    self._ear_low_start = now
                if self._ear_low_start is not None:
                    info["ear_duration"] = now - self._ear_low_start
                    ear_drowsy = info["ear_duration"] >= self.ear_duration_th
            else:
                self._ear_low_start = None
                self._ear_low_frames = 0
        else:
            self._ear_low_start = None
            self._ear_low_frames = 0

        # 2) Posture bad
        posture_drowsy = False
        if posture_data is not None:
            yaw = posture_data.get("head_side_angle")
            pitch = posture_data.get("head_updown_angle")
            shoulder = posture_data.get("shoulder_tilt")
            dist_cm = posture_data.get("eye_distance")

            posture_bad = False
            for val, limit in (
                (yaw, self.max_head_yaw),
                (pitch, self.max_head_pitch),
                (shoulder, self.max_shoulder_tilt),
            ):
                if val is not None and abs(val) > limit:
                    posture_bad = True
            if dist_cm is not None and (
                dist_cm < self.min_gaze_distance_cm or dist_cm > self.max_gaze_distance_cm
            ):
                posture_bad = True

            if posture_bad:
                if self._posture_bad_start is None:
                    self._posture_bad_start = now
                dur = now - self._posture_bad_start
                info["posture_bad_duration"] = dur
                posture_drowsy = dur >= self.posture_window_sec
            else:
                self._posture_bad_start = None

        # 3) Gaze-off
        gaze_drowsy = False
        if posture_data is not None:
            dist_cm = posture_data.get("eye_distance")
            if dist_cm is None or dist_cm < self.min_gaze_distance_cm:
                # Nếu mất khoảng cách quá lâu → reset timer
                if dist_cm is None and self._gaze_off_start is not None:
                    if now - self._gaze_off_start > self.MAX_MISSING_DIST_SEC:
                        self._gaze_off_start = None
                elif dist_cm is not None and dist_cm < self.min_gaze_distance_cm:
                    if self._gaze_off_start is None:
                        self._gaze_off_start = now
                    off_dur = now - self._gaze_off_start
                    info["gaze_off_duration"] = off_dur
                    gaze_drowsy = off_dur >= self.gaze_off_threshold_sec
            else:
                self._gaze_off_start = None

        # 4) Tổng hợp + hysteresis
        drowsy_signals = sum([ear_drowsy, posture_drowsy, gaze_drowsy])

        if drowsy_signals >= 2:
            self._drowsy = True
            self._drowsy_end_time = None
        else:
            if self._drowsy and self._drowsy_end_time is None:
                self._drowsy_end_time = now
            if self._drowsy_end_time and now - self._drowsy_end_time >= self.DROWSY_RELEASE_SEC:
                self._drowsy = False
                self._drowsy_end_time = None

        # 5) Ghi lý do
        if self._drowsy:
            if ear_drowsy and posture_drowsy:
                info["reason"] = "EAR + Posture"
            elif ear_drowsy and gaze_drowsy:
                info["reason"] = "EAR + GazeOff"
            elif posture_drowsy and gaze_drowsy:
                info["reason"] = "Posture + GazeOff"
            else:
                info["reason"] = "Combined"

        info["drowsiness_detected"] = self._drowsy
        return info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._ear_buf.clear()
        self._ear_low_start = None
        self._ear_low_frames = 0
        self._posture_bad_start = None
        self._gaze_off_start = None
        self._gaze_last_seen = None
        self._drowsy = False
        self._drowsy_end_time = None

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_ear(val: float, buf: deque) -> float:
        buf.append(val)
        if len(buf) < 2:
            return val
        weights = np.linspace(0.5, 1.0, len(buf))
        weights /= weights.sum()
        return float(np.average(buf, weights=weights))

    def reload_threshold(self, config_path: str = "config/settings.json"):
        cfg = get_config(config_path)
        self.ear_th = float(cfg["DROWSY_THRESHOLD"])