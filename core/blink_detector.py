from __future__ import annotations
import time
from typing import Tuple, Optional, Any

from utils.config import get_config
from core.eye_tracker import EyeTracker


class BlinkDetector:
    def __init__(
        self,
        config_path: str = "config/settings.json",
        eye_tracker: Optional[EyeTracker] = None,
    ):
        self.cfg = get_config(config_path)
        self.eye_tracker = eye_tracker

        # --- Tham số từ settings.json ---
        self.consecutive_frames = int(self.cfg["consecutive_frames"])  # số frame nhắm liên tiếp để tính là blink
        self.ear_th = float(self.cfg["BLINK_THRESHOLD"])
        self.max_blink_dur = float(self.cfg["max_blink_duration"])
        self.min_blink_gap = float(self.cfg["min_blink_interval"])
        self.max_head_yaw = float(self.cfg["max_head_side_angle"])
        self.max_head_pitch = float(self.cfg["max_head_updown_angle"])

        # --- Trạng thái runtime ---
        self.blink_count = 0
        self._closed_frames = 0
        self._blink_start = 0.0
        self._last_blink_ts = 0.0

        # --- Queue chống nhiễu khi quay đầu ---
        self._yaw_queue: list[float] = []
        self._yaw_window = 1.0  # giây
        self._counting_active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self) -> dict[str, Any]:
        """
        Gọi từ Manager hoặc vòng lặp chính.
        Trả về dict kết quả gọn gàng cho GUI / ChartManager / Database.
        """
        data = self.eye_tracker.get_latest() if self.eye_tracker else {}
        frame = data.get("frame")
        avg_ear = data.get("avg_ear")
        head_pitch = data.get("head_pitch")
        head_yaw = data.get("head_yaw")
        avg_contrast = data.get("avg_contrast")

        blink, info = self._detect(avg_ear, avg_contrast, head_pitch, head_yaw)

        info.update({
            "frame": frame,
            "blink_detected": blink,
            "total_blinks": self.blink_count,
            "ear": avg_ear,
        })
        return info

    # ------------------------------------------------------------------
    # Blink Detection Logic
    # ------------------------------------------------------------------
    def _detect(
        self,
        ear: float | None,
        contrast: float | None,
        pitch: float | None,
        yaw: float | None,
    ) -> Tuple[bool, dict[str, Any]]:
        now = time.time()
        blinked = False

        # --- 1. Loại bỏ nếu đầu nghiêng quá mức ---
        if pitch is not None and abs(pitch) > self.max_head_pitch:
            self._reset_state()
            return False, {"reason": "head_pitch_exceeded"}

        if yaw is not None:
            self._yaw_queue.append(now)
            self._yaw_queue = [t for t in self._yaw_queue if now - t <= self._yaw_window]
            if abs(yaw) > self.max_head_yaw and len(self._yaw_queue) > 5:
                self._reset_state()
                return False, {"reason": "head_yaw_exceeded"}

        # --- 2. Không có EAR hoặc nhiễu sáng mạnh ---
        if ear is None:
            self._reset_state()
            return False, {"reason": "no_ear"}

        if contrast is not None and contrast > 20.0:
            self._reset_state()
            return False, {"reason": "contrast_too_high"}

        # --- 3. FSM Blink Detection ---
        if ear < self.ear_th:
            if self._closed_frames == 0:
                self._blink_start = now
            self._closed_frames += 1
        else:
            if self._closed_frames >= self.consecutive_frames:
                duration = now - self._blink_start
                if duration <= self.max_blink_dur and (now - self._last_blink_ts) >= self.min_blink_gap:
                    blinked = True
                    if self._counting_active:
                        self.blink_count += 1
                    self._last_blink_ts = now
            self._closed_frames = 0

        return blinked, {"closed_frames": self._closed_frames}

    def _reset_state(self) -> None:
        self._closed_frames = 0

    # ------------------------------------------------------------------
    # Tiện ích ngoài
    # ------------------------------------------------------------------
    def reset_counter(self) -> None:
        self.blink_count = 0

    def get_count(self) -> int:
        return self.blink_count

    def start_counting(self) -> None:
        self._counting_active = True

    def stop_counting(self) -> None:
        self._counting_active = False
