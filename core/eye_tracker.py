from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from pathlib import Path
from typing import Dict, Any

from utils.config import get_config
from Execution.executor import ExecutorService
from utils.data_handler import save_data


class EyeTracker:
    _executor = ExecutorService(max_workers=2)

    def __init__(self, config_path: str | Path = "config/settings.json"):
        self.cfg = get_config(str(config_path))

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=float(self.cfg["min_detection_confidence"]),
            min_tracking_confidence=float(self.cfg["min_tracking_confidence"]),
        )

        self._cap: cv2.VideoCapture | None = None
        self._camera_idx = int(self.cfg.get("camera_index", 0))
        self._frame_rate = int(self.cfg.get("frame_rate", 30))

        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._running = False

        self._LEFT_EYE = self.cfg["LEFT_EYE"]
        self._RIGHT_EYE = self.cfg["RIGHT_EYE"]
        self._EPS = float(self.cfg["EPSILON"])
        self.f = None

    def start(self) -> None:
        if self._running:
            return
        self._cap = cv2.VideoCapture(self._camera_idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_idx}")
        self._cap.set(cv2.CAP_PROP_FPS, self._frame_rate)
        self._running = True
        EyeTracker._executor.submit(self._capture_loop)

    def stop(self) -> None:
        self._running = False
        EyeTracker._executor.shutdown()
        self._cleanup()

    def get_frame(self):
        return self.f

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return self._latest.copy()

    def _capture_loop(self) -> None:
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue

            self.f = frame.copy()
            data = self._process_frame(frame)

            with self._lock:
                self._latest = data

    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        h, w, _ = frame.shape
        out = {
            "frame": frame,
            "landmarks": None,
            "left_eye": None,
            "right_eye": None,
            "gaze_point": None,
            "left_ear": None,
            "right_ear": None,
            "avg_ear": None,
            "timestamp": time.time(),
            "proc_ms": 0.0,
        }

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            l_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in self._LEFT_EYE])
            r_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in self._RIGHT_EYE])

            out["left_contrast"] = self._eye_contrast(frame, l_pts)
            out["right_contrast"] = self._eye_contrast(frame, r_pts)
            contrasts = [v for v in (out["left_contrast"], out["right_contrast"]) if v is not None]
            out["avg_contrast"] = sum(contrasts) / len(contrasts) if contrasts else None

            out["left_eye"] = l_pts.tolist()
            out["right_eye"] = r_pts.tolist()
            out["gaze_point"] = np.mean([l_pts.mean(axis=0), r_pts.mean(axis=0)], axis=0).tolist()
            out["landmarks"] = [(p.x, p.y, p.z) for p in lm]

            if len(l_pts) == 6:
                out["left_ear"] = self._ear_improved(l_pts)
            if len(r_pts) == 6:
                out["right_ear"] = self._ear_improved(r_pts)

            ears = [e for e in (out["left_ear"], out["right_ear"]) if e is not None]
            out["avg_ear"] = (sum(ears) / len(ears)) if ears else None

        out["proc_ms"] = (time.perf_counter() - t0) * 1000
        return out

    @staticmethod
    def _ear_improved(pts: np.ndarray) -> float | None:
        if pts is None or len(pts) != 6:
            return None

        horizontal = np.linalg.norm(pts[0] - pts[3])
        if horizontal < 1e-6:
            return None

        vertical_1 = np.linalg.norm(pts[1] - pts[5])
        vertical_2 = np.linalg.norm(pts[2] - pts[4])
        middle_upper = (pts[1] + pts[2]) / 2
        middle_lower = (pts[4] + pts[5]) / 2
        vertical_3 = np.linalg.norm(middle_upper - middle_lower)

        ear = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
        return ear

    def _eye_contrast(self, frame: np.ndarray, eye_pts: np.ndarray) -> float | None:
        if eye_pts is None or len(eye_pts) != 6:
            return None

        x_min = int(min(p[0] for p in eye_pts))
        x_max = int(max(p[0] for p in eye_pts))
        y_min = int(min(p[1] for p in eye_pts))
        y_max = int(max(p[1] for p in eye_pts))

        if x_max - x_min < 2 or y_max - y_min < 2:
            return None

        roi = frame[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def _cleanup(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._face_mesh.close()

    def save_debug(self, path: str | Path) -> None:
        data = self.get_latest()
        img = data.pop("frame", None)
        save_data(data, str(path))
        if img is not None:
            cv2.imwrite(str(Path(path).with_suffix(".jpg")), img)
