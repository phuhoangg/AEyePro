from __future__ import annotations
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple

import mediapipe as mp
from utils.config import get_config
from utils.calibration import get_camera_calibration


class PostureAnalyzer:
    """
    Tách biệt, thread-safe posture analyzer.
    Tự động đọc toàn bộ tham số từ settings.json.
    Có thể chạy độc lập hoặc kết hợp EyeTracker.
    """

    def __init__(self, config_path: str = "config/settings.json"):
        cfg = get_config(config_path)

        # --- MediaPipe ---
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=float(cfg["pose_detection_confidence"]),
            min_tracking_confidence=float(cfg["pose_tracking_confidence"]),
        )

        # --- Cấu hình động ---
        self._focal = float(cfg["camera_focal_length"] or get_camera_calibration()["focal_length"])
        self._avg_eye_cm = float(cfg["AVERAGE_EYE_DISTANCE_CM"])
        self._min_eye_px = int(cfg["MIN_EYE_PIXEL_DISTANCE"])
        self._min_dist_cm = float(cfg["MIN_REASONABLE_DISTANCE"])
        self._max_dist_cm = float(cfg["MAX_REASONABLE_DISTANCE"])
        self._eps = float(cfg["EPSILON"])

        # --- Ngưỡng posture ---
        self._max_head_yaw = float(cfg["max_head_side_angle"])
        self._max_head_pitch = float(cfg["max_head_updown_angle"])
        self._max_shoulder_tilt = float(cfg["max_shoulder_tilt"])

        # --- Bộ lọc trượt đơn giản ---
        self._yaw_filter = deque(maxlen=5)
        self._pitch_filter = deque(maxlen=5)
        self._shoulder_filter = deque(maxlen=5)
        self._dist_filter = deque(maxlen=3)

        # --- Runtime state ---
        self._latest: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Trả về dict an toàn cho Manager / ChartManager / EncryptedStorage.
        Không rotate 180° (bỏ nếu camera đúng chiều).
        """
        if frame is None:
            return self._empty_result()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_landmarks:
            return self._empty_result()

        lm = results.pose_landmarks.landmark

        # Trích xuất landmark
        left_eye = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.LEFT_EYE, w, h)
        right_eye = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.RIGHT_EYE, w, h)
        left_shoulder = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, w, h)
        right_shoulder = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        nose = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.NOSE, w, h)

        # Góc
        eye_vec = right_eye[:2] - left_eye[:2]
        shoulder_vec = right_shoulder[:2] - left_shoulder[:2]
        head_vec = nose[:2] - (left_eye[:2] + right_eye[:2]) / 2

        yaw = self._angle(eye_vec, (1, 0))
        pitch = self._angle(head_vec, (0, -1))
        shoulder_tilt = self._angle(shoulder_vec, (1, 0))

        # Khoảng cách
        eye_px = np.linalg.norm(right_eye[:2] - left_eye[:2])
        distance_cm = None
        if eye_px >= self._min_eye_px:
            distance_cm = (self._avg_eye_cm * self._focal) / eye_px
            distance_cm = np.clip(distance_cm, self._min_dist_cm, self._max_dist_cm)

        # Lọc trượt
        self._yaw_filter.append(yaw)
        self._pitch_filter.append(pitch)
        self._shoulder_filter.append(shoulder_tilt)
        if distance_cm is not None:
            self._dist_filter.append(distance_cm)

        yaw_f = np.mean(self._yaw_filter)
        pitch_f = np.mean(self._pitch_filter)
        shoulder_f = np.mean(self._shoulder_filter)
        dist_f = np.mean(self._dist_filter) if self._dist_filter else None

        # Trạng thái posture
        status = self._classify(yaw_f, pitch_f, shoulder_f, dist_f)

        self._latest = {
            "timestamp": time.time(),
            "head_side_angle": yaw_f,
            "head_updown_angle": pitch_f,
            "shoulder_tilt": shoulder_f,
            "eye_distance": dist_f,
            "status": status,
        }
        return self._latest

    def get_latest(self) -> Dict[str, Any]:
        return self._latest.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _landmark_xyz(lm, enum, w, h) -> np.ndarray:
        p = lm[enum.value]
        return np.array([p.x * w, p.y * h, p.z * w])

    @staticmethod
    def _angle(v1: np.ndarray, v2: Tuple[float, float]) -> float:
        denom = max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-7)
        cos_ang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_ang)))

    def _classify(self, yaw: float, pitch: float, shoulder: float, dist: Optional[float]) -> str:
        # Đảm bảo góc luôn là góc nhỏ nhất so với trục chuẩn (0 hoặc 180)
        def min_angle(a):
            return min(abs(a), abs(180 - abs(a)))
        yaw_val = min_angle(yaw)
        pitch_val = min_angle(pitch)
        shoulder_val = min_angle(shoulder)
        if yaw_val > self._max_head_yaw:
            return "poor"
        if pitch_val > self._max_head_pitch:
            return "poor"
        if shoulder_val > self._max_shoulder_tilt:
            return "poor"
        if dist is not None and (dist < self._min_dist_cm or dist > self._max_dist_cm):
            return "poor"
        return "good"

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "head_side_angle": None,
            "head_updown_angle": None,
            "shoulder_tilt": None,
            "eye_distance": None,
            "status": "unknown",
        }

    # ------------------------------------------------------------------
    # Resource
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._pose.close()