from __future__ import annotations
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from utils.config import get_config

try:
    import subprocess
    _HAS_POWERSHELL = True
except Exception:
    _HAS_POWERSHELL = False


def normalize_angle(angle: Optional[float], threshold: float = 90.0) -> float:
    if angle is None:
        return 0.0
    if angle > 180 - threshold:
        return 180 - angle
    return angle


class HealthAssessor:
    def __init__(self, config_path: str = "config/settings.json"):
        cfg = get_config(config_path)
        self.session_duration_threshold = float(cfg["session_duration_threshold"])
        self.min_env_brightness = int(cfg["min_env_brightness"])
        self._scr_cache: Optional[int] = None
        self._scr_ts = 0.0
        self._scr_ttl = 10.0
        self.session_start = time.time()

    def assess(
        self,
        frame: Optional[np.ndarray],
        blink_count: int,
        posture: Dict[str, Any],
        drowsy_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        ts = time.time()
        session_dur = ts - self.session_start

        # Chuẩn hóa góc đầu/vai
        head_updown = normalize_angle(posture.get("head_updown_angle"))
        head_side = normalize_angle(posture.get("head_side_angle"))
        shoulder_tilt = normalize_angle(posture.get("shoulder_tilt"))

        return {
            "timestamp": ts,
            "session_duration": session_dur,
            "blink_count": blink_count,
            "ear": drowsy_info.get("average_ear"),
            "drowsiness_detected": drowsy_info.get("drowsiness_detected", False),
            "head_side_angle": head_side,
            "head_updown_angle": head_updown,
            "shoulder_tilt": shoulder_tilt,
            "eye_distance_cm": posture.get("eye_distance") or 0.0,
            "posture_status": posture.get("status", "unknown"),
            "screen_brightness": self._get_screen_brightness(),
            "env_brightness": self._get_env_brightness(frame),
            "long_session": session_dur > self.session_duration_threshold,
            "low_env_light": (self._get_env_brightness(frame) or 255) < self.min_env_brightness,
        }

    def reset(self) -> None:
        self.session_start = time.time()
        self._scr_cache = None
        self._scr_ts = 0.0

    def _get_screen_brightness(self) -> Optional[int]:
        if not _HAS_POWERSHELL:
            return None
        now = time.time()
        if self._scr_cache is not None and now - self._scr_ts < self._scr_ttl:
            return self._scr_cache
        try:
            out = subprocess.run(
                ["powershell", "-Command",
                 "(Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightness).CurrentBrightness"],
                capture_output=True, text=True, timeout=2,
            )
            val = int(out.stdout.strip().splitlines()[-1])
            self._scr_cache, self._scr_ts = val, now
            return val
        except Exception:
            return None

    @staticmethod
    def _get_env_brightness(frame: Optional[np.ndarray]) -> Optional[float]:
        if frame is None:
            return None
        h, w = frame.shape[:2]
        roi = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))