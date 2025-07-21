import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict
import threading
import time
from plyer import notification
import queue

from core.alert_engine import AlertEngine


class AlertGUI:
    def __init__(self, config_path: str = "config/settings.json", eye_tracker=None, blink_detector=None, posture_analyzer=None, drowsiness_detector=None, health_assessor=None):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.attributes("-topmost", True)

        # Biến lưu trạng thái
        self.snooze_end_time = 0
        self.desktop_end_time = 0

        # Buffer gộp cảnh báo
        self._buffer: List[Dict[str, str]] = []
        self._popup_win = None
        self._buffer_lock = threading.Lock()

        # Hàng đợi để xử lý cảnh báo trong luồng chính
        self._alert_queue = queue.Queue()

        self.engine = AlertEngine(
            on_alert=self.handle,
            config_path=config_path,
            eye_tracker=eye_tracker,
            blink_detector=blink_detector,
            posture_analyzer=posture_analyzer,
            drowsiness_detector=drowsiness_detector,
            health_assessor=health_assessor
        )

        # Kiểm tra hàng đợi trong luồng chính
        self._check_queue()

    # -----------------------------------------------------------
    # 1. Xử lý cảnh báo từ AlertEngine
    # -----------------------------------------------------------
    def handle(self, alerts: List[Dict[str, str]]) -> None:
        with self._buffer_lock:
            self._buffer.extend(alerts)
        # Đưa vào hàng đợi để xử lý trong luồng chính
        self._alert_queue.put(True)

    # -----------------------------------------------------------
    # 2. Kiểm tra hàng đợi và xử lý buffer
    # -----------------------------------------------------------
    def _check_queue(self):
        try:
            while True:
                # Thử lấy dữ liệu từ hàng đợi mà không chặn
                self._alert_queue.get_nowait()
                self._process_buffer()
        except queue.Empty:
            pass
        # Lên lịch kiểm tra tiếp theo
        self.root.after(100, self._check_queue)

    # -----------------------------------------------------------
    # 3. Gộp và hiển thị duy nhất 1 lần
    # -----------------------------------------------------------
    def _process_buffer(self):
        now = time.time()
        with self._buffer_lock:
            if not self._buffer:
                return
            alerts = self._buffer.copy()
            self._buffer.clear()

        if now < self.snooze_end_time:
            return  # Đang tắt
        if now < self.desktop_end_time:
            self._show_desktop_notif(alerts)
            return
        self._show_single_popup(alerts)

    # -----------------------------------------------------------
    # 4. Desktop notification (gộp)
    # -----------------------------------------------------------
    def _show_desktop_notif(self, alerts: List[Dict[str, str]]):
        has_high = any(a.get("priority") == "high" for a in alerts)
        title = "🚨 CẢNH BÁO NGHIÊM TRỌNG!" if has_high else "⚠️ Cảnh Báo Sức Khỏe"
        body = "\n".join(f"- {a['message']}" for a in alerts)
        notification.notify(
            title=title,
            message=body,
            app_name="AlertApp",
            timeout=10
        )

    # -----------------------------------------------------------
    # 5. Popup duy nhất, căn giữa
    # -----------------------------------------------------------
    def _show_single_popup(self, alerts: List[Dict[str, str]]):
        # Đóng popup cũ nếu đang mở
        if self._popup_win and self._popup_win.winfo_exists():
            self._popup_win.destroy()

        has_high = any(a.get("priority") == "high" for a in alerts)
        title = "🚨 CẢNH BÁO NGHIÊM TRỌNG!" if has_high else "⚠️ Cảnh Báo Sức Khỏe"
        messages = "\n".join(f"- {a['message']}" for a in alerts)

        self._popup_win = tk.Toplevel(self.root)
        self._popup_win.title("Cảnh báo")
        self._popup_win.resizable(False, False)
        self._popup_win.attributes("-topmost", True)
        self._popup_win.grab_set()

        # Căn giữa màn hình với kích thước lớn hơn
        self._popup_win.update_idletasks()
        w, h = 600, 300
        x = (self._popup_win.winfo_screenwidth() - w) // 2
        y = (self._popup_win.winfo_screenheight() - h) // 2
        self._popup_win.geometry(f"{w}x{h}+{x}+{y}")

        tk.Label(self._popup_win, text=title, font=("Segoe UI", 18, "bold"), fg="red").pack(pady=(20, 10))
        tk.Label(self._popup_win, text=messages, justify="center", wraplength=560, font=("Segoe UI", 14)).pack(pady=(0, 20))

        btn_frame = tk.Frame(self._popup_win)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="OK", command=self._popup_win.destroy).grid(row=0, column=0, padx=10)

        snooze_mb = tk.Menubutton(btn_frame, text="Tắt thông báo", relief="raised")
        snooze_mb.grid(row=0, column=1, padx=10)
        snooze_menu = tk.Menu(snooze_mb, tearoff=0)
        for m in (5, 15, 60):
            snooze_menu.add_command(label=f"{m} phút",
                                    command=lambda mins=m: self._snooze_and_close(mins))
        snooze_mb["menu"] = snooze_menu

        style_mb = tk.Menubutton(btn_frame, text="Thay đổi kiểu thông báo", relief="raised")
        style_mb.grid(row=0, column=2, padx=10)
        style_menu = tk.Menu(style_mb, tearoff=0)
        for m in (10, 30, 120):
            style_menu.add_command(label=f"{m} phút",
                                   command=lambda mins=m: self._switch_to_desktop(mins))
        style_mb["menu"] = style_menu

    # -----------------------------------------------------------
    # 6. Tắt / chuyển kiểu thông báo
    # -----------------------------------------------------------
    def _snooze_and_close(self, minutes: int):
        if self._popup_win and self._popup_win.winfo_exists():
            self._popup_win.destroy()
        self.snooze_end_time = time.time() + minutes * 60
        self.desktop_end_time = 0
        messagebox.showinfo("Tạm dừng", f"Tắt thông báo trong {minutes} phút.")
        threading.Timer(minutes * 60, self._restore_normal_mode).start()

    def _switch_to_desktop(self, minutes: int):
        if self._popup_win and self._popup_win.winfo_exists():
            self._popup_win.destroy()
        self.desktop_end_time = time.time() + minutes * 60
        self.snooze_end_time = 0
        messagebox.showinfo("Chế độ mới", f"Chuyển sang thông báo desktop trong {minutes} phút.")
        threading.Timer(minutes * 60, self._restore_normal_mode).start()

    def _restore_normal_mode(self):
        self.snooze_end_time = 0
        self.desktop_end_time = 0

    # -----------------------------------------------------------
    # 7. Chạy ứng dụng
    # -----------------------------------------------------------
    def run(self) -> None:
        self.engine.start()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.engine.stop()

    def update_alert_state(self, health_data):
        # Gọi từ GUI/mainloop để cập nhật dữ liệu mới nhất cho AlertEngine
        self.engine.update_state({"health_data": health_data})


# ----------------------------------------------------------------------
if __name__ == "__main__":
    AlertGUI().run()