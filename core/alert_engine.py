from __future__ import annotations
import time
import threading
from typing import Callable

from utils.config import get_config
from core.eye_tracker import EyeTracker
from core.blink_detector import BlinkDetector
from core.posture_analyzer import PostureAnalyzer
from core.drowsiness_detector import DrowsinessDetector
from core.health_assessor import HealthAssessor


class AlertEngine:

    def __init__(
        self,
        on_alert: Callable[[dict], None],
        config_path: str = "config/settings.json",
        eye_tracker: EyeTracker = None,
        blink_detector: BlinkDetector = None,
        posture_analyzer: PostureAnalyzer = None,
        drowsiness_detector: DrowsinessDetector = None,
        health_assessor: HealthAssessor = None,
    ):
        self.cfg = get_config(config_path)
        self.on_alert = on_alert

        # Pipeline: dùng instance truyền vào nếu có, không thì tạo mới
        self.eye_tracker = eye_tracker or EyeTracker(config_path)
        self.blink_detector = blink_detector or BlinkDetector(config_path, eye_tracker=self.eye_tracker)
        self.posture_analyzer = posture_analyzer or PostureAnalyzer(config_path)
        self.drowsiness_detector = drowsiness_detector or DrowsinessDetector(config_path)
        self.health_assessor = health_assessor or HealthAssessor(config_path)

        # Trạng thái cảnh báo
        self.drowsy_start_time: float | None = None
        self.drowsy_duration_threshold = float(self.cfg.get("drowsy_duration_threshold", 3.0))
        self.is_drowsy_alerted = False
        self.last_drowsy_state = False

        self.general_start_time: float | None = None
        self.general_is_alerted = False
        self.last_general_alerts: set[str] = set()

        # Luồng
        self._running = False
        self._loop_thread: threading.Thread | None = None
        # State cập nhật từ GUI/mainloop
        self._latest_state = None
        self._state_lock = threading.Lock()

    # ----------------- Public -----------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Không start EyeTracker ở đây, chỉ dùng get_latest
        self._loop_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._loop_thread.start()
        print("🚀 AlertEngine đã khởi động...")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
        # Không stop EyeTracker ở đây

    def update_state(self, state: dict):
        # state: dict chứa posture, drowsy_info, blink_count, health_data đã tính sẵn từ GUI
        with self._state_lock:
            self._latest_state = state.copy()

    # ----------------- Main loop -----------------
    def _main_loop(self) -> None:
        while self._running:
            with self._state_lock:
                state = self._latest_state.copy() if self._latest_state else None
            if not state:
                time.sleep(0.01)
                continue
            # Sử dụng trực tiếp health_data đã tính sẵn
            health = state.get("health_data")
            if not health:
                time.sleep(0.01)
                continue
            alerts = self._build_alerts(health)
            self._handle_alerts(alerts)
            time.sleep(1 / 30)

    # ------------- Logic build & handle alerts -------------
    def _build_alerts(self, data: dict) -> dict:
        drowsy_alerts = []
        if data.get("drowsiness_detected"):
            drowsy_alerts.append({"message": "😴 Có dấu hiệu buồn ngủ!", "priority": "high"})

        general_alerts = []

        if data.get("session_duration", 0) > self.health_assessor.session_duration_threshold * 0.8:
            general_alerts.append({"message": "🕒 Bạn đã làm việc quá lâu, hãy nghỉ ngơi!", "priority": "medium"})

        eye_distance = data.get("eye_distance_cm")
        if eye_distance is not None:
            if eye_distance < self.cfg.get("MIN_REASONABLE_DISTANCE", 30):
                general_alerts.append({"message": "📏 Mắt quá gần màn hình!", "priority": "medium"})
            elif eye_distance > self.cfg.get("MAX_REASONABLE_DISTANCE", 100):
                general_alerts.append({"message": "📏 Mắt quá xa màn hình!", "priority": "low"})

        tilt = data.get("head_updown_angle") or 0
        if abs(tilt) > self.cfg.get("max_head_updown_angle", 18):
            general_alerts.append({"message": "↕️ Đầu nghiêng lên/xuống quá nhiều!", "priority": "low"})

        side_angle = data.get("head_side_angle") or 0
        if abs(side_angle) > self.cfg.get("max_head_side_angle", 15):
            general_alerts.append({"message": "↔️ Đầu nghiêng sang trái/phải!", "priority": "low"})

        shoulder = data.get("shoulder_tilt") or 0
        if abs(shoulder) > self.cfg.get("max_shoulder_tilt", 12):
            general_alerts.append({"message": "🧍‍♂️ Vai lệch nhiều!", "priority": "low"})

        env_brightness = data.get("env_brightness") or 0
        screen_brightness = data.get("screen_brightness") or 0

        if env_brightness < self.cfg.get("min_env_brightness", 40):
            general_alerts.append({"message": "🌤 Môi trường quá tối!", "priority": "medium"})

        self._check_brightness_compatibility(screen_brightness, env_brightness, general_alerts)

        return {"drowsy": drowsy_alerts, "general": general_alerts}

    def _check_brightness_compatibility(self, screen_lux: float, env_lux: float, alerts: list):
        """
        screen_lux : độ sáng màn hình (lux hoặc nits)
        env_lux    : độ sáng môi trường (cùng đơn vị)
        alerts     : list để append dict {"message": str, "priority": "medium"}
        """

        # --- chuẩn hóa độ sáng màn hình về 0-100 % ---
        # Giả sử datasheet hoặc cấu hình đã chứa max_nits của màn hình
        MAX_NITS = 200
        screen_lux_norm = (screen_lux / 100.0) * MAX_NITS

        # --- tính chênh lệch ---
        diff = screen_lux_norm - env_lux
        abs_diff = abs(diff)

        def add(msg):
            alerts.append({"message": msg, "priority": "medium"})

        if abs_diff <= 30:
            return
        elif 30 < abs_diff <= 70:
            if diff > 0:
                add("👁️ Màn hình sáng hơn môi trường, hãy giảm độ sáng hoặc bật thêm đèn.")
            else:
                add("👁️ Môi trường sáng hơn màn hình, hãy tăng độ sáng màn hình.")
        else:
            if diff > 0:
                add("⚠️ Màn hình quá sáng so với xung quanh, hãy giảm ngay!")
            else:
                add("⚠️ Môi trường quá sáng so với màn hình, hãy tăng độ sáng màn hình!")

    def _handle_alerts(self, alerts: dict) -> None:
        drowsy_alerts = alerts.get("drowsy", [])
        general_alerts = alerts.get("general", [])

        show_drowsy = self._should_show_drowsy_alert(drowsy_alerts)
        show_general = self._should_show_general_alert(general_alerts)

        all_alerts = []
        if show_drowsy:
            all_alerts.extend(drowsy_alerts)
        if show_general:
            all_alerts.extend(general_alerts)

        if all_alerts:
            self.on_alert(all_alerts)

    # ----------------- Logic show alert -----------------
    def _should_show_drowsy_alert(self, drowsy_alerts):
        now = time.time()
        current = bool(drowsy_alerts)

        if not current:
            self.last_drowsy_state = False
            return None
        else:
            if not self.last_drowsy_state:
                self.drowsy_start_time = now
                self.last_drowsy_alert_time = now
                self.last_drowsy_state = True
                return True
            else:
                if now - getattr(self, 'last_drowsy_alert_time', 0) >= 2:
                    self.last_drowsy_alert_time = now
                    return True
                else:
                    return False

    def _should_show_general_alert(self, general_alerts):
        now = time.time()

        if not general_alerts:
            self.general_start_time = None
            self.general_is_alerted = False
            self.last_general_alerts = set()
            self.last_general_alert_time = 0
            return False

        current = set(a["message"] for a in general_alerts)

        if current != self.last_general_alerts:
            self.last_general_alerts = current
            self.general_start_time = now
            self.general_is_alerted = False
            return False

        if self.general_start_time is not None and now - self.general_start_time >= self.drowsy_duration_threshold:
            if now - getattr(self, "last_general_alert_time", 0) >= 5:
                self.general_is_alerted = True
                self.last_general_alert_time = now
                return True

        return False