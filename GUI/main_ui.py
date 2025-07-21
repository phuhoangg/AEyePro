from __future__ import annotations
import customtkinter as ctk
import threading
import logging
from datetime import datetime
import webbrowser
from Manager import chart_manager
import pandas as pd
import glob
from pathlib import Path

from GUI.monitoring_view import MonitoringView
from GUI.summary_view import SummaryView
from GUI.ai_view import AIAssistantView
from Manager.alert_manager import AlertGUI
from core.health_data_collector import HealthDataCollector

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EyeMonitoringApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("AEyeProo")
        self.root.geometry("1280x720")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Load dữ liệu 1 lần cho toàn app
        self.df_summary = pd.read_csv('data/summary.csv') if Path('data/summary.csv').exists() else pd.DataFrame()
        realtime_files = glob.glob('data/realtime_*.csv')
        dfs = []
        for f in realtime_files:
            try:
                df = pd.read_csv(f)
                df['session_id'] = Path(f).stem.split('_')[-1]
                dfs.append(df)
            except Exception as e:
                print(f"[ERROR] Lỗi đọc {f}: {e}")
        self.df_realtime = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.current_view = None
        self.views = {}
        self.switch_view("monitoring")

    def switch_view(self, module):
        # Ẩn view hiện tại
        if self.current_view is not None:
            self.views[self.current_view].parent.pack_forget()
        # Tạo mới nếu chưa có
        if module not in self.views:
            if module == "monitoring":
                frame = ctk.CTkFrame(self.main_frame)
                frame.pack(fill="both", expand=True)
                self.views[module] = MonitoringView(frame, self)
            elif module == "summary":
                frame = ctk.CTkFrame(self.main_frame)
                frame.pack(fill="both", expand=True)
                self.views[module] = SummaryView(frame, self, self.df_summary, self.df_realtime)
            elif module == "ai":
                frame = ctk.CTkFrame(self.main_frame)
                frame.pack(fill="both", expand=True)
                self.views[module] = AIAssistantView(frame, self)
            else:
                frame = ctk.CTkFrame(self.main_frame)
                frame.pack(fill="both", expand=True)
                ctk.CTkLabel(frame, text=f"Module '{module}' not implemented").pack()
                self.views[module] = frame
        else:
            self.views[module].parent.pack(fill="both", expand=True)
        self.current_view = module

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = EyeMonitoringApp()
    app.run()