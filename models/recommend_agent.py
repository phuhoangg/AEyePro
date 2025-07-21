import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
from core.health_data_collector import HealthDataCollector
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of recommendations"""
    POSTURE = "posture"
    EYE_CARE = "eye_care"
    BREAK_TIME = "break_time"
    LIGHTING = "lighting"
    EXERCISE = "exercise"
    ALERT = "alert"


class SeverityLevel(Enum):
    """Severity levels for recommendations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Health alert data structure"""
    id: str
    type: RecommendationType
    severity: SeverityLevel
    message: str
    recommendation: str
    timestamp: datetime
    is_active: bool = True
    data: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
            "is_active": self.is_active,
            "data": self.data
        }


@dataclass
class HealthThresholds:
    """Health thresholds for recommendations"""
    # Eye care thresholds
    min_blink_rate: float = 15.0  # blinks per minute
    max_eye_distance: float = 70.0  # cm
    min_eye_distance: float = 50.0  # cm

    # Posture thresholds
    max_head_tilt: float = 15.0  # degrees
    max_head_side_angle: float = 10.0  # degrees
    max_shoulder_tilt: float = 5.0  # degrees

    # Session thresholds
    max_session_duration: float = 60.0  # minutes
    break_reminder_interval: float = 30.0  # minutes

    # Lighting thresholds
    min_screen_brightness: float = 40.0  # percentage
    max_screen_brightness: float = 80.0  # percentage

    # Drowsiness thresholds
    min_ear_ratio: float = 0.25  # Eye Aspect Ratio
    max_drowsiness_count: int = 5  # in 5 minutes

    def update_from_dict(self, data: Dict[str, float]) -> None:
        """Update thresholds from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class HealthAnalyzer:
    """Analyzes health data and generates recommendations"""

    def __init__(self, thresholds: HealthThresholds):
        self.thresholds = thresholds
        self.active_alerts = {}
        self.recommendation_history = deque(maxlen=50)

    def analyze_health_data(self, current_data: Dict, history: List[Dict]) -> Tuple[List[HealthAlert], Dict]:
        """Analyze health data and generate recommendations"""
        if not current_data:
            return [], {"status": "no_data", "score": 0}

        alerts = []
        health_status = self._calculate_health_status(current_data, history)

        # Analyze different health aspects
        alerts.extend(self._analyze_eye_health(current_data, history))
        alerts.extend(self._analyze_posture(current_data))
        alerts.extend(self._analyze_session_duration(current_data))
        alerts.extend(self._analyze_lighting(current_data))
        alerts.extend(self._analyze_drowsiness(current_data))

        # Update active alerts
        self._update_active_alerts(alerts)

        return alerts, health_status

    def _calculate_health_status(self, current_data: Dict, history: List[Dict]) -> Dict:
        """Calculate overall health status"""
        scores = []

        # Eye health score (0-100)
        blink_rate = current_data.get('blink_count', 0) * 60 / max(1, len(history))
        eye_distance = current_data.get('avg_eye_distance', 60)

        eye_score = 100
        if blink_rate < self.thresholds.min_blink_rate:
            eye_score -= (self.thresholds.min_blink_rate - blink_rate) * 2
        if eye_distance < self.thresholds.min_eye_distance:
            eye_score -= (self.thresholds.min_eye_distance - eye_distance) * 1.5
        elif eye_distance > self.thresholds.max_eye_distance:
            eye_score -= (eye_distance - self.thresholds.max_eye_distance) * 1.5
        scores.append(max(0, eye_score))

        # Posture score (0-100)
        head_tilt = abs(current_data.get('avg_head_tilt', 0))
        head_side = abs(current_data.get('head_side_angle', 0))
        shoulder_tilt = abs(current_data.get('shoulder_tilt', 0))

        posture_score = 100
        if head_tilt > self.thresholds.max_head_tilt:
            posture_score -= (head_tilt - self.thresholds.max_head_tilt) * 2
        if head_side > self.thresholds.max_head_side_angle:
            posture_score -= (head_side - self.thresholds.max_head_side_angle) * 2
        if shoulder_tilt > self.thresholds.max_shoulder_tilt:
            posture_score -= (shoulder_tilt - self.thresholds.max_shoulder_tilt) * 3
        scores.append(max(0, posture_score))

        # Session duration score (0-100)
        session_duration = current_data.get('session_duration', 0) / 60
        session_score = 100
        if session_duration > self.thresholds.max_session_duration:
            session_score -= (session_duration - self.thresholds.max_session_duration) * 0.5
        scores.append(max(0, session_score))

        overall_score = sum(scores) / len(scores)

        if overall_score >= 90:
            status = "excellent"
        elif overall_score >= 70:
            status = "good"
        elif overall_score >= 50:
            status = "fair"
        else:
            status = "poor"

        return {
            "status": status,
            "score": round(overall_score, 1),
            "eye_score": round(scores[0], 1),
            "posture_score": round(scores[1], 1),
            "session_score": round(scores[2], 1),
            "blink_rate": round(blink_rate, 1),
            "eye_distance": round(eye_distance, 1),
            "session_duration_minutes": round(session_duration, 1)
        }

    def _analyze_eye_health(self, current_data: Dict, history: List[Dict]) -> List[HealthAlert]:
        """Analyze eye health and generate alerts"""
        alerts = []

        # Calculate blink rate
        blink_rate = current_data.get('blink_count', 0) * 60 / max(1, len(history))
        eye_distance = current_data.get('avg_eye_distance', 60)

        # Low blink rate alert
        if blink_rate < self.thresholds.min_blink_rate:
            severity = SeverityLevel.HIGH if blink_rate < 10 else SeverityLevel.MEDIUM
            alerts.append(HealthAlert(
                id="low_blink_rate",
                type=RecommendationType.EYE_CARE,
                severity=severity,
                message=f"Tần suất chớp mắt thấp: {blink_rate:.1f} lần/phút",
                recommendation="Hãy chớp mắt nhiều hơn hoặc nghỉ ngơi 5 phút để tránh khô mắt",
                timestamp=datetime.now(),
                data={"blink_rate": blink_rate}
            ))

        # Eye distance alerts
        if eye_distance < self.thresholds.min_eye_distance:
            severity = SeverityLevel.HIGH if eye_distance < 40 else SeverityLevel.MEDIUM
            alerts.append(HealthAlert(
                id="too_close_screen",
                type=RecommendationType.EYE_CARE,
                severity=severity,
                message=f"Khoảng cách màn hình quá gần: {eye_distance:.1f}cm",
                recommendation="Hãy ngồi xa màn hình hơn, khoảng cách lý tưởng là 50-70cm",
                timestamp=datetime.now(),
                data={"eye_distance": eye_distance}
            ))
        elif eye_distance > self.thresholds.max_eye_distance:
            alerts.append(HealthAlert(
                id="too_far_screen",
                type=RecommendationType.EYE_CARE,
                severity=SeverityLevel.LOW,
                message=f"Khoảng cách màn hình hơi xa: {eye_distance:.1f}cm",
                recommendation="Có thể ngồi gần màn hình hơn một chút để giảm căng thẳng mắt",
                timestamp=datetime.now(),
                data={"eye_distance": eye_distance}
            ))

        return alerts

    def _analyze_posture(self, current_data: Dict) -> List[HealthAlert]:
        """Analyze posture and generate alerts"""
        alerts = []

        head_tilt = current_data.get('avg_head_tilt', 0)
        head_side = current_data.get('head_side_angle', 0)
        shoulder_tilt = current_data.get('shoulder_tilt', 0)
        posture_status = current_data.get('posture_status', 'good')

        # Head tilt alerts
        if abs(head_tilt) > self.thresholds.max_head_tilt:
            severity = SeverityLevel.HIGH if abs(head_tilt) > 25 else SeverityLevel.MEDIUM
            direction = "cúi xuống" if head_tilt > 0 else "ngẩng lên"
            alerts.append(HealthAlert(
                id="head_tilt",
                type=RecommendationType.POSTURE,
                severity=severity,
                message=f"Đầu {direction} quá nhiều: {abs(head_tilt):.1f}°",
                recommendation="Hãy giữ đầu thẳng và điều chỉnh độ cao màn hình",
                timestamp=datetime.now(),
                data={"head_tilt": head_tilt}
            ))

        # Head side angle alerts
        if abs(head_side) > self.thresholds.max_head_side_angle:
            direction = "trái" if head_side > 0 else "phải"
            alerts.append(HealthAlert(
                id="head_side_tilt",
                type=RecommendationType.POSTURE,
                severity=SeverityLevel.MEDIUM,
                message=f"Đầu nghiêng {direction}: {abs(head_side):.1f}°",
                recommendation="Hãy giữ đầu thẳng và đặt màn hình chính diện",
                timestamp=datetime.now(),
                data={"head_side_angle": head_side}
            ))

        # Shoulder tilt alerts
        if abs(shoulder_tilt) > self.thresholds.max_shoulder_tilt:
            alerts.append(HealthAlert(
                id="shoulder_tilt",
                type=RecommendationType.POSTURE,
                severity=SeverityLevel.MEDIUM,
                message=f"Vai không cân bằng: {abs(shoulder_tilt):.1f}°",
                recommendation="Hãy thả lỏng vai và giữ hai vai ngang bằng nhau",
                timestamp=datetime.now(),
                data={"shoulder_tilt": shoulder_tilt}
            ))

        # General posture status
        if posture_status in ['forward', 'backward', 'tilt']:
            posture_messages = {
                'forward': "Bạn đang ngồi cúi về phía trước",
                'backward': "Bạn đang ngồi dựa lưng quá nhiều",
                'tilt': "Tư thế ngồi không cân bằng"
            }
            alerts.append(HealthAlert(
                id="general_posture",
                type=RecommendationType.POSTURE,
                severity=SeverityLevel.MEDIUM,
                message=posture_messages[posture_status],
                recommendation="Hãy ngồi thẳng lưng, hai chân đặt sát sàn",
                timestamp=datetime.now(),
                data={"posture_status": posture_status}
            ))

        return alerts

    def _analyze_session_duration(self, current_data: Dict) -> List[HealthAlert]:
        """Analyze session duration and generate break recommendations"""
        alerts = []
        session_duration = current_data.get('session_duration', 0) / 60

        if session_duration > self.thresholds.max_session_duration:
            severity = SeverityLevel.HIGH if session_duration > 90 else SeverityLevel.MEDIUM
            alerts.append(HealthAlert(
                id="long_session",
                type=RecommendationType.BREAK_TIME,
                severity=severity,
                message=f"Đã làm việc liên tục {session_duration:.1f} phút",
                recommendation="Hãy nghỉ ngơi 10-15 phút, đứng dậy đi lại và nhìn xa",
                timestamp=datetime.now(),
                data={"session_duration": session_duration}
            ))
        elif session_duration > self.thresholds.break_reminder_interval:
            alerts.append(HealthAlert(
                id="break_reminder",
                type=RecommendationType.BREAK_TIME,
                severity=SeverityLevel.LOW,
                message=f"Đã làm việc {session_duration:.1f} phút",
                recommendation="Hãy nghỉ ngơi 2-3 phút, chớp mắt và nhìn xa",
                timestamp=datetime.now(),
                data={"session_duration": session_duration}
            ))

        return alerts

    def _analyze_lighting(self, current_data: Dict) -> List[HealthAlert]:
        """Analyze lighting conditions and generate alerts"""
        alerts = []
        screen_brightness = current_data.get('screen_brightness', 50)
        env_brightness = current_data.get('env_brightness', 100)

        # Screen brightness alerts
        if screen_brightness < self.thresholds.min_screen_brightness:
            alerts.append(HealthAlert(
                id="low_screen_brightness",
                type=RecommendationType.LIGHTING,
                severity=SeverityLevel.MEDIUM,
                message=f"Độ sáng màn hình thấp: {screen_brightness:.1f}%",
                recommendation="Hãy tăng độ sáng màn hình lên 40-80%",
                timestamp=datetime.now(),
                data={"screen_brightness": screen_brightness}
            ))
        elif screen_brightness > self.thresholds.max_screen_brightness:
            alerts.append(HealthAlert(
                id="high_screen_brightness",
                type=RecommendationType.LIGHTING,
                severity=SeverityLevel.MEDIUM,
                message=f"Độ sáng màn hình cao: {screen_brightness:.1f}%",
                recommendation="Hãy giảm độ sáng màn hình xuống 40-80%",
                timestamp=datetime.now(),
                data={"screen_brightness": screen_brightness}
            ))

        # Environmental lighting
        if env_brightness < 80:
            alerts.append(HealthAlert(
                id="low_env_lighting",
                type=RecommendationType.LIGHTING,
                severity=SeverityLevel.LOW,
                message="Ánh sáng môi trường yếu",
                recommendation="Hãy bật thêm đèn hoặc mở rèm để tăng ánh sáng",
                timestamp=datetime.now(),
                data={"env_brightness": env_brightness}
            ))

        return alerts

    def _analyze_drowsiness(self, current_data: Dict) -> List[HealthAlert]:
        """Analyze drowsiness and generate alerts"""
        alerts = []
        drowsiness = current_data.get('drowsiness', False)
        ear_ratio = current_data.get('ear', 0.3)

        if drowsiness or ear_ratio < self.thresholds.min_ear_ratio:
            severity = SeverityLevel.HIGH if ear_ratio < 0.2 else SeverityLevel.MEDIUM
            alerts.append(HealthAlert(
                id="drowsiness_detected",
                type=RecommendationType.ALERT,
                severity=severity,
                message="Phát hiện dấu hiệu buồn ngủ",
                recommendation="Hãy nghỉ ngơi, uống nước, hoặc đi lại để tỉnh táo hơn",
                timestamp=datetime.now(),
                data={"drowsiness": drowsiness, "ear_ratio": ear_ratio}
            ))

        return alerts

    def _update_active_alerts(self, new_alerts: List[HealthAlert]):
        """Update active alerts"""
        # Mark existing alerts as inactive
        for alert in self.active_alerts.values():
            alert.is_active = False

        # Add new alerts
        for alert in new_alerts:
            self.active_alerts[alert.id] = alert
            self.recommendation_history.append(alert)

    def get_active_alerts(self, severity_filter: Optional[str] = None,
                          type_filter: Optional[str] = None) -> List[HealthAlert]:
        """Get active alerts with optional filtering"""
        alerts = [alert for alert in self.active_alerts.values() if alert.is_active]

        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity.value == severity_filter]
        if type_filter:
            alerts = [alert for alert in alerts if alert.type.value == type_filter]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def clear_alerts(self):
        """Clear all alerts"""
        self.active_alerts.clear()
        self.recommendation_history.clear()


class RecommendAgent:
    """Main recommendation engine for desktop application"""

    def __init__(self, health_data_collector=HealthDataCollector):
        """
        Initialize RecommendAgent with health data collector

        Args:
            health_data_collector: Instance of HealthDataCollector from core module
        """
        self.thresholds = HealthThresholds()
        self.analyzer = HealthAnalyzer(self.thresholds)
        self.health_data_collector = health_data_collector
        self.is_running = False
        self.analysis_thread = None
        self.analysis_interval = 2.0  # seconds
        self.data_history = deque(maxlen=300)  # Keep 5 minutes of data (300 seconds)
        logger.info("Recommendation Agent initialized")

    def set_health_data_collector(self, health_data_collector):
        """Set or update the health data collector"""
        self.health_data_collector = health_data_collector
        logger.info("Health data collector updated")

    def start(self):
        """Start recommendation engine"""
        if self.is_running:
            logger.warning("Recommendation Agent already running")
            return

        if not self.health_data_collector:
            logger.error("No health data collector provided")
            return

        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        logger.info("Recommendation Agent started")

    def stop(self):
        """Stop recommendation engine"""
        if not self.is_running:
            return

        self.is_running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        logger.info("Recommendation Agent stopped")

    def _analysis_loop(self):
        """Main analysis loop"""
        while self.is_running:
            try:
                self._update_data_history()
                time.sleep(self.analysis_interval)
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(self.analysis_interval)

    def _update_data_history(self):
        """Update data history from health data collector"""
        if not self.health_data_collector:
            return

        # Get current accumulated data
        current_data = getattr(self.health_data_collector, 'accumulated_data', {})
        if not current_data:
            return

        # Add session duration if not present
        if 'session_duration' not in current_data:
            session_start = getattr(self.health_data_collector, 'session_start_time', time.time())
            current_data['session_duration'] = time.time() - session_start

        # Add to history
        data_with_timestamp = current_data.copy()
        data_with_timestamp['timestamp'] = time.time()
        self.data_history.append(data_with_timestamp)

    def _get_current_data(self) -> Optional[Dict]:
        """Get current health data"""
        if not self.health_data_collector:
            return None

        current_data = getattr(self.health_data_collector, 'accumulated_data', {})
        if not current_data:
            return None

        # Add session duration
        session_start = getattr(self.health_data_collector, 'session_start_time', time.time())
        current_data['session_duration'] = time.time() - session_start

        return current_data

    def _get_history_data(self, minutes: int = 5) -> List[Dict]:
        """Get historical data for specified minutes"""
        if not self.data_history:
            return []

        cutoff_time = time.time() - (minutes * 60)
        return [data for data in self.data_history if data.get('timestamp', 0) > cutoff_time]

    def get_recommendations(self, include_analysis: bool = False,
                            severity_filter: Optional[str] = None,
                            type_filter: Optional[str] = None,
                            max_results: int = 10) -> Dict[str, Any]:
        """Get current recommendations"""
        try:
            current_data = self._get_current_data()
            history = self._get_history_data(minutes=5)

            if not current_data:
                return {
                    "status": "no_data",
                    "message": "Không có dữ liệu sức khỏe hiện tại",
                    "recommendations": [],
                    "health_status": {},
                    "analysis": None
                }

            # Analyze health data
            alerts, health_status = self.analyzer.analyze_health_data(current_data, history)
            active_alerts = self.analyzer.get_active_alerts(severity_filter, type_filter)[:max_results]

            return {
                "status": "success",
                "message": f"Tìm thấy {len(active_alerts)} khuyến nghị",
                "recommendations": [alert.to_dict() for alert in active_alerts],
                "health_status": health_status,
                "analysis": self._generate_analysis(health_status, active_alerts) if include_analysis else None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                "status": "error",
                "message": f"Lỗi khi tạo khuyến nghị: {str(e)}",
                "recommendations": [],
                "health_status": {},
                "analysis": None
            }

    def _generate_analysis(self, health_status: Dict, alerts: List[HealthAlert]) -> str:
        """Generate simple analysis"""
        score = health_status.get('score', 0)

        if score >= 90:
            base_message = "Tình trạng sức khỏe rất tốt! "
        elif score >= 70:
            base_message = "Tình trạng sức khỏe khá tốt. "
        elif score >= 50:
            base_message = "Tình trạng sức khỏe ở mức trung bình. "
        else:
            base_message = "Tình trạng sức khỏe cần cải thiện. "

        if not alerts:
            return base_message + "Hãy tiếp tục duy trì thói quen tốt."

        high_priority = [a for a in alerts if a.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]

        if high_priority:
            base_message += f"Có {len(high_priority)} vấn đề cần chú ý ngay. "

        return base_message + "Hãy thường xuyên nghỉ ngơi và giữ tư thế tốt."

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        current_data = self._get_current_data()
        history = self._get_history_data(minutes=5)

        if not current_data:
            return {"status": "no_data", "message": "Không có dữ liệu"}

        _, health_status = self.analyzer.analyze_health_data(current_data, history)
        return health_status

    def get_active_alerts(self, severity_filter: Optional[str] = None,
                          type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts as dictionaries"""
        alerts = self.analyzer.get_active_alerts(severity_filter, type_filter)
        return [alert.to_dict() for alert in alerts]

    def get_alerts_by_type(self, alert_type: str) -> List[Dict[str, Any]]:
        """Get alerts filtered by type"""
        return self.get_active_alerts(type_filter=alert_type)

    def get_alerts_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get alerts filtered by severity"""
        return self.get_active_alerts(severity_filter=severity)

    def reset_session(self):
        """Reset session timer in health data collector"""
        if self.health_data_collector and hasattr(self.health_data_collector, 'session_start_time'):
            self.health_data_collector.session_start_time = time.time()
            self.data_history.clear()
            logger.info("Session reset")

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update health thresholds"""
        self.thresholds.update_from_dict(new_thresholds)
        logger.info("Health thresholds updated")

    def clear_all_alerts(self):
        """Clear all alerts"""
        self.analyzer.clear_alerts()
        logger.info("All alerts cleared")

    def is_engine_running(self) -> bool:
        """Check if engine is running"""
        return self.is_running

    def get_data_collector_status(self) -> Dict[str, Any]:
        """Get status of health data collector"""
        if not self.health_data_collector:
            return {"status": "not_connected", "message": "No health data collector"}

        return {
            "status": "connected",
            "running": getattr(self.health_data_collector, 'running', False),
            "session_id": getattr(self.health_data_collector, 'session_id', 'unknown'),
            "total_records": getattr(self.health_data_collector, 'total_records', 0),
            "history_length": len(self.data_history)
        }


