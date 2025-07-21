from __future__ import annotations
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from Execution.executor import ExecutorService
from utils.config import get_config
from utils.data_handler import DATA_DIR, append_csv_row
logger = logging.getLogger(__name__)


class HealthDataCollector:
    """
    - realtime_YYYYMMDD_<session_id>.csv : lưu realtime (1 row / giây)
    - summary.csv : lưu summary (1 row / session)
    """

    def __init__(
        self,
        collect_interval: float = 1.0,
        config_path: str = "config/settings.json",
        executor: Optional[ExecutorService] = None,
    ):
        cfg = get_config(config_path)
        self.collect_interval = collect_interval
        self.session_id = str(uuid.uuid4())[:8]
        self.executor = executor or ExecutorService(max_workers=2)

        self.data_dir = DATA_DIR
        self.rt_csv_path = None
        self.summary_csv_path = self.data_dir / "summary.csv"

        self._running = False
        self._future = None
        self._start_ts = 0.0
        self._latest: Dict[str, Any] = {}
        self._reset_stats()

    # ----------------------------- public ----------------------------- #
    def start_collection(self) -> None:
        if self._running:
            logger.warning("Already running")
            return
        self._running = True
        self._start_ts = time.time()
        self._reset_stats()
        date_str = datetime.now().strftime("%Y%m%d")
        self.rt_csv_path = self.data_dir / f"realtime_{date_str}_{self.session_id}.csv"
        self._future = self.executor.submit(self._loop)
        logger.info("HealthDataCollector started (%s)", self.session_id)

    def stop_collection(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._future:
            self._future.result(timeout=2.0)
        self._write_summary()
        logger.info("HealthDataCollector stopped (%s)", self.session_id)

    def update_health_data(self, health_data: Dict[str, Any]) -> None:
        # Đếm số lần chớp mắt và buồn ngủ dựa trên trạng thái chuyển từ False -> True
        blink_now = bool(health_data.get("blink_detected", False))
        drowsy_now = bool(health_data.get("drowsiness_detected", False))
        if blink_now and not self._last_blink_state:
            self.blink_count += 1
        if drowsy_now and not self._last_drowsy_state:
            self.drowsiness_count += 1
        self._last_blink_state = blink_now
        self._last_drowsy_state = drowsy_now
        self._latest = health_data.copy()

    # ----------------------------- internal --------------------------- #
    def _reset_stats(self):
        self.total_records = 0
        self.total_blinks = 0
        self.sum_dist = 0.0
        self.sum_head_tilt = 0.0
        self.sum_head_side = 0.0
        self.sum_shoulder_angle = 0.0
        self.sum_screen_brightness = 0.0
        self.sum_ambient_brightness = 0.0
        self.drowsiness_count = 0
        self.bad_posture_count = 0
        self.blink_count = 0
        self._last_blink_state = False
        self._last_drowsy_state = False

    def _loop(self):
        while self._running:
            if self._latest:
                self._collect_once()
            time.sleep(self.collect_interval)

    def _collect_once(self):
        ts = time.time()
        blink = self.blink_count
        if not isinstance(blink, int) or blink < 0 or blink > 10000:
            blink = 0
        dist = float(self._latest.get("eye_distance_cm", 0.0))
        tilt = float(self._latest.get("head_updown_angle", 0.0))
        side = float(self._latest.get("head_side_angle", 0.0))
        shoulder = float(self._latest.get("shoulder_tilt", 0.0))
        screen = int(self._latest.get("screen_brightness", 0))
        ambient = float(self._latest.get("env_brightness", 0.0))
        drowsy   = self.drowsiness_count
        posture = str(self._latest.get("posture_status", "neutral"))
        date_str = datetime.fromtimestamp(ts).strftime("%d/%m/%Y")
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        # Ghi vào file CSV realtime
        row = {
            "ID": self.session_id,
            "date": date_str,
            "time": time_str,
            "blink_count": blink,
            "distance": dist,
            "head_tilt": tilt,
            "head_side": side,
            "shoulder_angle": shoulder,
            "screen_br": screen,
            "ambient_br": ambient,
            "drowsiness_detected": drowsy,
            "posture_status": posture,
        }
        fieldnames = [
            "ID", "date", "time", "blink_count", "distance", "head_tilt", "head_side", "shoulder_angle", "screen_br", "ambient_br", "drowsiness_detected", "posture_status"
        ]
        append_csv_row(row, self.rt_csv_path, fieldnames=fieldnames)
        # accumulate summary
        self.total_records += 1
        self.total_blinks = blink
        self.sum_dist += dist
        self.sum_head_tilt += tilt
        self.sum_head_side += side
        self.sum_shoulder_angle += shoulder
        self.sum_screen_brightness += screen
        self.sum_ambient_brightness += ambient
        if posture == "poor":
            self.bad_posture_count += 1

    def _write_summary(self):
        if self.total_records == 0:
            return
        duration = time.time() - self._start_ts
        duration_minutes = max(duration / 60.0, 1e-6)
        blink_per_min = self.total_blinks / duration_minutes
        def safe_avg(total):
            return total / self.total_records if self.total_records else 0.0
        row = {
            "session_id": self.session_id,
            "begin_timestamp": datetime.fromtimestamp(self._start_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "end_timestamp": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
            "blink_per_minute": round(blink_per_min, 1),
            "avg_distance": round(safe_avg(self.sum_dist), 2),
            "avg_head_tilt": round(safe_avg(self.sum_head_tilt), 2),
            "avg_head_side": round(safe_avg(self.sum_head_side), 2),
            "avg_shoulder_angle": round(safe_avg(self.sum_shoulder_angle), 2),
            "avg_screen_brightness": round(safe_avg(self.sum_screen_brightness), 1),
            "avg_ambient_brightness": round(safe_avg(self.sum_ambient_brightness), 2),
            "number_of_drowsiness": self.drowsiness_count,
            "bad_posture_count": self.bad_posture_count,
            "session_duration_seconds": round(duration, 1),
        }
        fieldnames = [
            "session_id", "begin_timestamp", "end_timestamp", "blink_per_minute", "avg_distance", "avg_head_tilt", "avg_head_side", "avg_shoulder_angle", "avg_screen_brightness", "avg_ambient_brightness", "number_of_drowsiness", "bad_posture_count", "session_duration_seconds"
        ]
        append_csv_row(row, self.summary_csv_path, fieldnames=fieldnames)