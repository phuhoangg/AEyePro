import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from core.eye_tracker import EyeTracker
from core.blink_detector import BlinkDetector
from core.health_assessor import HealthAssessor
from core.health_data_collector import HealthDataCollector
from core.posture_analyzer import PostureAnalyzer
from core.drowsiness_detector import DrowsinessDetector
from Manager.alert_manager import AlertGUI
from utils.config import get_config, save_config
from Execution.executor import ExecutorService
from utils.data_handler import delete_csv_file


class MonitoringView:
    def __init__(self, parent, app, config_path: str = "config/settings.json"):
        self.app = app
        self.cfg = get_config(config_path)
        self.config_path = config_path
        self.parent = parent
        self.user_name = self.cfg.get("user_name", "User")

        # Backend
        self.executor = ExecutorService(max_workers=4)
        self.eye_tracker = EyeTracker(config_path)
        self.blink_detector = BlinkDetector(config_path, self.eye_tracker)
        self.health_assessor = HealthAssessor(config_path)
        self.health_collector = HealthDataCollector(config_path=config_path, executor=self.executor)
        self.posture_analyzer = PostureAnalyzer(config_path)
        self.drowsiness_detector = DrowsinessDetector(config_path)
        self.alert_gui = AlertGUI(
            config_path,
            eye_tracker=self.eye_tracker,
            blink_detector=self.blink_detector,
            posture_analyzer=self.posture_analyzer,
            drowsiness_detector=self.drowsiness_detector,
            health_assessor=self.health_assessor
        )
        self.ear_threshold = float(self.cfg.get("BLINK_THRESHOLD", 0.25))
        self.calibrating = False
        self.monitoring = False
        self.calibration_ears = []
        self.overlay_face_grid = tk.BooleanVar(value=True)
        self.overlay_eye_box = tk.BooleanVar(value=True)
        self.log_lines = []
        self._stop_event = threading.Event()
        self._latest_health_data = {}
        self._monitoring_thread = None
        self._monitoring_active = threading.Event()
        self._db_storage = None
        self._exiting = False
        self._last_backend_frame_ts = None
        self._session_start_time = None
        self._session_duration = 0

        self._setup_ui()
        self.eye_tracker.start()
        self._update_camera()

    def _setup_ui(self):
        self.parent.grid_rowconfigure(1, weight=1)
        self.parent.grid_columnconfigure(0, weight=0)
        self.parent.grid_columnconfigure(1, weight=1)
        self.parent.grid_columnconfigure(2, weight=0)

        # Menu button (hamburger) - tăng kích thước, trigger bằng click
        self.menu_btn = ctk.CTkButton(self.parent, text="☰", width=60, height=60, corner_radius=12, font=ctk.CTkFont(size=28, weight="bold"))
        self.menu_btn.grid(row=0, column=0, sticky="nw", padx=(10, 0), pady=10)
        self.menu_btn.bind("<Button-1>", self._show_menu)
        self.menu_popup = None

        # Thông tin realtime bên trái camera
        self.info_frame = ctk.CTkFrame(self.parent, width=260, corner_radius=12)
        self.info_frame.grid(row=1, column=0, sticky="nsw", padx=(10, 0), pady=10)
        ctk.CTkLabel(self.info_frame, text="Realtime Health Info", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 10))
        self.info_labels = {}
        metrics = [
            ("EAR", "ear"),
            ("Head Up/Down", "head_updown_angle"),
            ("Head Side", "head_side_angle"),
            ("Shoulder Tilt", "shoulder_tilt"),
            ("Eye Distance", "eye_distance_cm"),
            ("Posture", "posture_status"),
            ("Session Time", "session_duration"),
            ("Drowsy", "drowsiness_detected"),
        ]
        for label, key in metrics:
            row = ctk.CTkFrame(self.info_frame)
            row.pack(fill="x", pady=2, padx=10)
            ctk.CTkLabel(row, text=label+":", width=120, anchor="w").pack(side="left")
            self.info_labels[key] = ctk.CTkLabel(row, text="N/A", width=100, anchor="w")
            self.info_labels[key].pack(side="left")

        # Camera frame (dịch sang phải)
        self.camera_frame = ctk.CTkFrame(self.parent, corner_radius=16, fg_color="#fff")
        self.camera_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.camera_frame.grid_rowconfigure(0, weight=1)
        self.camera_frame.grid_columnconfigure(0, weight=1)
        self.cam_label = ctk.CTkLabel(self.camera_frame, text="", width=640, height=480, corner_radius=12)
        self.cam_label.grid(row=0, column=0, padx=10, pady=10)

        # Overlay toggles
        overlay_frame = ctk.CTkFrame(self.camera_frame, fg_color="#f5f5f5", corner_radius=10)
        overlay_frame.grid(row=1, column=0, pady=(0, 10), sticky="ew")
        ctk.CTkCheckBox(overlay_frame, text="Face Grid", variable=self.overlay_face_grid, command=self._force_update).pack(side="left", padx=10)
        ctk.CTkCheckBox(overlay_frame, text="Eye Box", variable=self.overlay_eye_box, command=self._force_update).pack(side="left", padx=10)

        # Tên user dưới camera
        self.user_label = ctk.CTkLabel(self.camera_frame, text=f"User: {self.user_name}", font=ctk.CTkFont(size=14))
        self.user_label.grid(row=2, column=0, pady=(0, 10))

        # Sidebar phải (logs)
        self.rightbar = ctk.CTkFrame(self.parent, width=220, corner_radius=12)
        self.rightbar.grid(row=1, column=2, sticky="nse", padx=(0, 10), pady=10)
        ctk.CTkLabel(self.rightbar, text="Activity Log", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))
        self.log_box = ctk.CTkTextbox(self.rightbar, width=200, height=400, font=ctk.CTkFont(size=12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_box.insert("end", "Monitoring log initialized...\n")
        self.log_box.configure(state="disabled")

        # Nút chức năng dưới camera
        btn_frame = ctk.CTkFrame(self.parent, fg_color="#f5f5f5", corner_radius=10)
        btn_frame.grid(row=2, column=1, pady=(0, 10), sticky="ew", padx=10)
        self.calib_btn = ctk.CTkButton(btn_frame, text="Cập nhật EAR", command=self._calibrate, corner_radius=8)
        self.calib_btn.pack(side="left", padx=10, pady=10)
        self.monitor_btn = ctk.CTkButton(btn_frame, text="Start Monitoring", command=self._toggle_monitoring, corner_radius=8)
        self.monitor_btn.pack(side="left", padx=10, pady=10)
        # Nút Exit
        self.exit_btn = ctk.CTkButton(btn_frame, text="Exit", command=self._exit_app, corner_radius=8, fg_color="#e53935", text_color="white")
        self.exit_btn.pack(side="left", padx=10, pady=10)

        # Hiển thị ngưỡng EAR hiện tại dưới camera
        self.ear_th_label = ctk.CTkLabel(self.parent, text=f"EAR trung bình: N/A | Drowsiness Threshold: {self.drowsiness_detector.ear_th:.3f}", font=ctk.CTkFont(size=14, weight="bold"))
        self.ear_th_label.grid(row=3, column=1, pady=(0, 10))

        # Thanh tiến trình calibrate
        self.progress_bar = ctk.CTkProgressBar(self.parent, width=400)
        self.progress_bar.grid(row=4, column=1, pady=(0, 20))
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()

    def _show_menu(self, event=None):
        if self.menu_popup and self.menu_popup.winfo_exists():
            self.menu_popup.lift()
            return
        x = self.menu_btn.winfo_rootx()
        y = self.menu_btn.winfo_rooty() + self.menu_btn.winfo_height()
        self.menu_popup = tk.Toplevel(self.parent)
        self.menu_popup.title("Menu")
        self.menu_popup.geometry(f"220x220+{x}+{y}")
        self.menu_popup.transient(self.parent)
        self.menu_popup.overrideredirect(True)
        self.menu_popup.configure(bg="#f0f0f0", highlightthickness=2, highlightbackground="#888")
        try:
            self.menu_popup.tk.call("wm", "attributes", self.menu_popup._w, "-topmost", True)
            self.menu_popup.tk.call("wm", "attributes", self.menu_popup._w, "-alpha", 0.98)
        except Exception:
            pass
        ctk.CTkLabel(self.menu_popup, text="Chọn module", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkButton(self.menu_popup, text="Monitoring", command=lambda: self._switch_module("monitoring")).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.menu_popup, text="AI Assistant", command=lambda: self._switch_module("ai")).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.menu_popup, text="Summary", command=lambda: self._switch_module("summary")).pack(fill="x", padx=20, pady=5)
        self.menu_popup.bind("<FocusOut>", lambda e: self.menu_popup.destroy())
        self.menu_popup.focus_set()

    def _switch_module(self, module):
        self._log(f"Switch to module: {module}")
        if self.menu_popup and self.menu_popup.winfo_exists():
            self.menu_popup.destroy()
        if hasattr(self.app, "switch_view"):
            self.app.switch_view(module)

    def _update_camera(self):
        frame_data = self.eye_tracker.get_latest()
        frame = frame_data.get("frame")
        landmarks = frame_data.get("landmarks")
        left_eye = frame_data.get("left_eye")
        right_eye = frame_data.get("right_eye")
        avg_ear = frame_data.get("avg_ear")
        frame_ts = frame_data.get("timestamp")

        # Xử lý realtime info trực tiếp trên frame mới nhất (30Hz)
        blink_info = self.blink_detector.update()
        blink_count = self.blink_detector.get_count()
        blink_detected = blink_info.get("blink_detected", False)
        if blink_detected:
            self._log(f"Blink detected! Total: {blink_count}")
        posture = self.posture_analyzer.analyze(frame) if frame is not None else {}
        drowsy_info = self.drowsiness_detector.update(ear=avg_ear, posture_data=posture)
        # Session time chỉ tính khi monitoring
        if self.monitoring:
            if self._session_start_time is None:
                self._session_start_time = frame_ts or time.time()
            session_duration = (frame_ts or time.time()) - self._session_start_time
        else:
            session_duration = self._session_duration if self._session_duration else 0
        # Luôn tạo health_data ở ngoài if/else
        health_data = self.health_assessor.assess(
            frame=frame,
            blink_count=blink_count,
            posture=posture,
            drowsy_info={
                "average_ear": avg_ear,
                "drowsiness_detected": drowsy_info.get("drowsiness_detected", False)
            }
        )
        health_data["session_duration"] = session_duration
        # Thêm trường blink_detected vào health_data để health_collector đếm đúng
        health_data["blink_detected"] = blink_detected
        self._latest_health_data = health_data.copy()
        # Cập nhật alert state cho AlertEngine (nếu đang monitoring)
        if self.monitoring:
            self.alert_gui.update_alert_state(health_data)
        for key, label in self.info_labels.items():
            value = health_data.get(key)
            if value is not None:
                if key == "drowsiness_detected":
                    label.configure(text="😴" if value else "😊")
                if key in ["head_updown_angle", "head_side_angle", "shoulder_tilt"]:
                    label.configure(text=f"{value:.1f}°")
                elif key == "ear":
                    label.configure(text=f"{value:.2f}")
                elif key == "eye_distance_cm":
                    label.configure(text=f"{value:.1f} cm")
                elif key == "session_duration":
                    label.configure(text=f"{value:.0f} s")
                else:
                    label.configure(text=str(value))
            else:
                label.configure(text="N/A")

        if frame is not None:
            disp_frame = frame.copy()
            # Overlay grid
            if self.overlay_face_grid.get() and landmarks:
                h, w, _ = disp_frame.shape
                for (x, y, z) in landmarks:
                    cv2.circle(disp_frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)
            # Overlay eye box
            if self.overlay_eye_box.get():
                eye_color = (30, 136, 229)  # #1E88E5 in BGR (blue)
                for eye_pts in [left_eye, right_eye]:
                    if eye_pts:
                        pts = [(int(x), int(y)) for x, y in eye_pts]
                        cv2.polylines(disp_frame, [np.array(pts, dtype=np.int32)], isClosed=True, color=eye_color, thickness=2)
            img = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            ctk_img = ctk.CTkImage(light_image=img.resize((640, 480)), size=(640, 480))
            self.cam_label.configure(image=ctk_img)
            self.cam_label.image = ctk_img
        self.parent.after(33, self._update_camera)  # ~30Hz

        # Nếu đang calibrate, lưu EAR
        if self.calibrating and avg_ear is not None:
            self.calibration_ears.append(avg_ear)

        # Nếu đang monitoring, luôn submit backend mỗi frame
        if self.monitoring:
            health_data_copy = health_data.copy()
            try:
                self.executor.submit(self.health_collector.update_health_data, health_data_copy)
            except RuntimeError:
                pass

    def _calibrate(self):
        if self.calibrating:
            return
        self.calibrating = True
        self.calibration_ears = []
        self._log("Calibration started: Please keep eyes open for 10 seconds...")
        self.calib_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_bar.grid()
        self._calib_start_time = time.time()
        self._update_progress_bar()
        self.parent.after(10000, self._finish_calibration)

    def _update_progress_bar(self):
        if not self.calibrating:
            self.progress_bar.grid_remove()
            return
        elapsed = time.time() - self._calib_start_time
        progress = min(elapsed / 10.0, 1.0)
        self.progress_bar.set(progress)
        if progress < 1.0:
            self.parent.after(100, self._update_progress_bar)
        else:
            self.progress_bar.set(1.0)

    def _finish_calibration(self):
        if self.calibration_ears:
            avg = sum(self.calibration_ears) / len(self.calibration_ears)
            self.ear_threshold = avg * 0.85
            self._log(f"Calibration done.")
            self.ear_th_label.configure(text=f"EAR: {avg:.3f} | Drowsiness Threshold: {self.drowsiness_detector.ear_th:.3f}")
            # Cập nhật lại ngưỡng BLINK_THRESHOLD và DROWSY_THRESHOLD trong config và lưu file
            self.cfg["BLINK_THRESHOLD"] = self.ear_threshold
            self.cfg["DROWSY_THRESHOLD"] = self.ear_threshold
            save_config(self.cfg, self.config_path)
            self.drowsiness_detector.reload_threshold(self.config_path)
        else:
            self._log("Calibration failed: No EAR data.")
        self.calibrating = False
        self.calib_btn.configure(state="normal")
        self.progress_bar.grid_remove()

    def _toggle_monitoring(self):
        self.monitoring = not self.monitoring
        if self.monitoring:
            self.monitor_btn.configure(text="Stop Monitoring")
            self._log("Monitoring started.")
            self._session_start_time = None
            self._session_duration = 0
            self._last_backend_frame_ts = None
            self.blink_detector.reset_counter()  # Reset blink count cho phiên mới
            self.blink_detector.start_counting() # Bắt đầu đếm blink
            self._start_backend_modules()
        else:
            self.monitor_btn.configure(text="Start Monitoring")
            self._log("Monitoring stopped.")
            self.blink_detector.stop_counting() # Dừng đếm blink
            self._stop_backend_modules()

    def _start_backend_modules(self):
        self.health_collector.start_collection()
        self.alert_gui.engine.start()
        self._monitoring_active.set()
        # Đảm bảo không tạo nhiều thread
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()

    def _stop_backend_modules(self):
        self._monitoring_active.clear()
        self.health_collector.stop_collection()
        self.alert_gui.engine.stop()
        # Đợi thread monitoring dừng hẳn
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=1)
        self._monitoring_thread = None
        # Log lại cho user biết đã ghi summary
        self._log("Session summary written to summary.csv.")

    def _monitoring_loop(self):
        # Không cần xử lý backend ở đây nữa, mọi thứ đã được submit từ mainloop
        pass

    def _confirm(self):
        self._log("Confirm action pressed.")
        messagebox.showinfo("Confirm", "Action confirmed!")

    def _cancel(self):
        self._log("Cancel action pressed.")
        messagebox.showinfo("Cancel", "Action cancelled!")

    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _force_update(self):
        pass

    def secure_delete_summary(self):
        # Xóa file summary an toàn
        csv_path = Path("data/summary.csv")
        delete_csv_file(csv_path)
        self._log("Summary CSV securely deleted.")

    def shutdown(self):
        # Đảm bảo mã hóa lại file khi tắt app
        if self.health_collector:
            self.health_collector.sum_storage.close()
        if self._db_storage:
            self._db_storage.close()
        self.executor.shutdown()

    def _exit_app(self):
        self._exiting = True
        # Nếu đang monitoring thì dừng lại
        self._monitoring_active.clear()
        if self.monitoring:
            self.monitoring = False
            self._stop_backend_modules()
        # Đợi thread monitoring dừng hẳn trước khi shutdown executor
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=2)
        try:
            if hasattr(self, 'shutdown'):
                self.shutdown()
        except Exception:
            pass
        # Không dừng camera (EyeTracker), chỉ destroy window
        self.parent.winfo_toplevel().destroy()
