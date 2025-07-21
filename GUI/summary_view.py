import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns  # Thêm seaborn để vẽ heatmap

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pathlib import Path
from datetime import datetime, timedelta
from Manager.chart_manager import ChartManager
import threading
import webbrowser
from tkcalendar import DateEntry


class SummaryView:
    def __init__(self, parent, app, df_summary, df_realtime):
        self.app = app
        self.parent = parent
        self.df_summary = df_summary
        self.df_realtime = ChartManager.clean_data(df_realtime)
        self.selected_charts = {
            "blink": tk.BooleanVar(value=True),
            "drowsy": tk.BooleanVar(value=True),
            "posture": tk.BooleanVar(value=True),
            "focus": tk.BooleanVar(value=True),
            "session": tk.BooleanVar(value=True),
        }
        self.menu_popup = None
        self._setup_ui()
        self.refresh_dashboard()

    def _setup_ui(self):
        self.parent.grid_rowconfigure(3, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)

        # Header frame with gradient-like background - compact
        header_frame = ctk.CTkFrame(self.parent, corner_radius=8, height=50, fg_color=("#f0f0f0", "#2b2b2b"))
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)

        # Menu button (hamburger) - compact design
        self.menu_btn = ctk.CTkButton(
            header_frame,
            text="☰",
            width=35,
            height=35,
            corner_radius=18,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=("#3498db", "#2980b9"),
            hover_color=("#2980b9", "#3498db")
        )
        self.menu_btn.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.menu_btn.bind("<Button-1>", self._show_menu)

        # Title with better styling - compact
        title = ctk.CTkLabel(
            header_frame,
            text="📊 Summary Dashboard",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#2c3e50", "#ecf0f1")
        )
        title.grid(row=0, column=1, pady=8, sticky="")

        # Filter frame with improved styling - compact
        filter_frame = ctk.CTkFrame(self.parent, corner_radius=8, fg_color=("#f8f9fa", "#34495e"))
        filter_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        filter_frame.grid_columnconfigure(5, weight=1)

        # Date filter components with compact spacing
        ctk.CTkLabel(filter_frame, text="📅 Từ:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0,
                                                                                                padx=(10, 5), pady=8)
        self.from_date = DateEntry(filter_frame, width=12, background='#3498db', foreground='white', borderwidth=1,
                                   date_pattern='yyyy-mm-dd')
        self.from_date.grid(row=0, column=1, padx=5, pady=8)

        ctk.CTkLabel(filter_frame, text="📅 Đến:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=2,
                                                                                                 padx=(10, 5), pady=8)
        self.to_date = DateEntry(filter_frame, width=12, background='#3498db', foreground='white', borderwidth=1,
                                 date_pattern='yyyy-mm-dd')
        self.to_date.grid(row=0, column=3, padx=5, pady=8)

        today = datetime.now().date()
        self.from_date.set_date(today - timedelta(days=6))
        self.to_date.set_date(today)

        # Filter buttons with compact styling
        apply_btn = ctk.CTkButton(
            filter_frame,
            text="🔍 Apply",
            command=self.refresh_dashboard,
            corner_radius=6,
            width=80,
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#27ae60", "#229954"),
            hover_color=("#229954", "#27ae60")
        )
        apply_btn.grid(row=0, column=4, padx=8, pady=8)

        all_data_btn = ctk.CTkButton(
            filter_frame,
            text="📊 All Data",
            command=self._show_all_data,
            corner_radius=6,
            width=100,
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#e74c3c", "#c0392b"),
            hover_color=("#c0392b", "#e74c3c")
        )
        all_data_btn.grid(row=0, column=5, padx=(8, 10), pady=8, sticky="e")

        # Info cards frame - compact and centered
        self.info_cards_frame = ctk.CTkFrame(self.parent, corner_radius=8, fg_color="transparent")
        self.info_cards_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.info_cards_frame.grid_columnconfigure(0, weight=1)
        self.info_cards_frame.grid_columnconfigure(1, weight=1)
        self.info_cards_frame.grid_columnconfigure(2, weight=1)

        # Dashboard charts area with vertical scrollbar - maximized space
        charts_outer = ctk.CTkFrame(self.parent, corner_radius=8, fg_color=("#ffffff", "#2b2b2b"))
        charts_outer.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        charts_outer.grid_rowconfigure(0, weight=1)
        charts_outer.grid_columnconfigure(0, weight=1)

        self.charts_canvas = tk.Canvas(charts_outer, borderwidth=0, highlightthickness=0)
        self.charts_canvas.grid(row=0, column=0, sticky="nsew")

        # Custom scrollbar styling
        scrollbar = ttk.Scrollbar(charts_outer, orient="vertical", command=self.charts_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.charts_canvas.configure(yscrollcommand=scrollbar.set)

        self.charts_frame = ctk.CTkFrame(self.charts_canvas, corner_radius=0, fg_color="transparent")
        self.charts_window = self.charts_canvas.create_window((0, 0), window=self.charts_frame, anchor="nw")
        self.charts_frame.grid_columnconfigure(0, weight=1)

        def _on_frame_configure(event):
            self.charts_canvas.configure(scrollregion=self.charts_canvas.bbox("all"))

        self.charts_frame.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(event):
            canvas_width = event.width
            self.charts_canvas.itemconfig(self.charts_window, width=canvas_width)

        self.charts_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self.charts_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.charts_canvas.bind("<MouseWheel>", _on_mousewheel)

    def _show_menu(self, event=None):
        if self.menu_popup and self.menu_popup.winfo_exists():
            self.menu_popup.lift()
            return
        x = self.menu_btn.winfo_rootx()
        y = self.menu_btn.winfo_rooty() + self.menu_btn.winfo_height()
        self.menu_popup = tk.Toplevel(self.parent)
        self.menu_popup.title("Menu")
        self.menu_popup.geometry(f"250x280+{x}+{y}")
        self.menu_popup.transient(self.parent)
        self.menu_popup.overrideredirect(True)
        self.menu_popup.configure(bg="#34495e", highlightthickness=2, highlightbackground="#3498db")

        try:
            self.menu_popup.tk.call("wm", "attributes", self.menu_popup._w, "-topmost", True)
            self.menu_popup.tk.call("wm", "attributes", self.menu_popup._w, "-alpha", 0.95)
        except Exception:
            pass

        # Menu header
        header_label = ctk.CTkLabel(
            self.menu_popup,
            text="🔧 Chọn module",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#ecf0f1"
        )
        header_label.pack(pady=(15, 10))

        # Menu buttons with icons
        menu_buttons = [
            ("📊 Monitoring", "monitoring"),
            ("🤖 AI Assistant", "ai"),
            ("📈 Summary", "summary")
        ]

        for text, module in menu_buttons:
            btn = ctk.CTkButton(
                self.menu_popup,
                text=text,
                command=lambda m=module: self._switch_module(m),
                width=200,
                height=40,
                corner_radius=8,
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color=("#3498db", "#2980b9"),
                hover_color=("#2980b9", "#3498db")
            )
            btn.pack(fill="x", padx=20, pady=8)

        self.menu_popup.bind("<FocusOut>", lambda e: self.menu_popup.destroy())
        self.menu_popup.focus_set()

    def _switch_module(self, module):
        if self.menu_popup and self.menu_popup.winfo_exists():
            self.menu_popup.destroy()
        if hasattr(self.app, "switch_view"):
            self.app.switch_view(module)

    def _show_all_data(self):
        # Xóa filter ngày, hiển thị toàn bộ dữ liệu
        self.from_date.set_date(datetime(1970, 1, 1))
        self.to_date.set_date(datetime.now().date())
        self.refresh_dashboard()

    def _create_info_card(self, parent, title, value, icon, color, row, column):
        """Create a compact styled info card"""
        card = ctk.CTkFrame(parent, corner_radius=8, fg_color=color)
        card.grid(row=row, column=column, sticky="ew", padx=5, pady=5)

        # Icon - smaller
        icon_label = ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=20))
        icon_label.pack(pady=(8, 2))

        # Value - compact
        value_label = ctk.CTkLabel(card, text=str(value), font=ctk.CTkFont(size=14, weight="bold"))
        value_label.pack(pady=1)

        # Title - compact
        title_label = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=10), wraplength=180)
        title_label.pack(pady=(1, 8))

        return card

    def refresh_dashboard(self):
        try:
            from_date = pd.to_datetime(self.from_date.get_date())
            to_date = pd.to_datetime(self.to_date.get_date()) + pd.Timedelta(days=1)
        except Exception:
            from_date = None
            to_date = None

        df_summary = self.df_summary.copy()
        df_realtime = self.df_realtime.copy()

        # Chuyển cột date và time thành timestamp trong df_realtime
        if not df_realtime.empty and 'date' in df_realtime.columns and 'time' in df_realtime.columns:
            df_realtime['timestamp'] = pd.to_datetime(df_realtime['date'] + ' ' + df_realtime['time'], format='%d/%m/%Y %H:%M:%S')
        elif not df_realtime.empty and 'timestamp' not in df_realtime.columns:
            print("Warning: 'timestamp' column not found and could not be created from 'date' and 'time' columns.")

        # Chuyển cột begin_timestamp về datetime trong df_summary
        if not df_summary.empty and 'begin_timestamp' in df_summary.columns:
            df_summary['begin_datetime'] = pd.to_datetime(df_summary['begin_timestamp'])

        # Lọc dữ liệu theo khoảng thời gian
        if not df_realtime.empty and 'timestamp' in df_realtime.columns and from_date and to_date and from_date.year > 1970:
            df_realtime = df_realtime[(df_realtime['timestamp'] >= from_date) & (df_realtime['timestamp'] < to_date)]
        if not df_summary.empty and 'begin_datetime' in df_summary.columns and from_date and to_date and from_date.year > 1970:
            df_summary = df_summary[
                (df_summary['begin_datetime'] >= from_date) & (df_summary['begin_datetime'] < to_date)]

        # Clear existing widgets
        for widget in self.info_cards_frame.winfo_children():
            widget.destroy()
        for widget in self.charts_frame.winfo_children():
            widget.destroy()

        # --- TÍNH TOÁN THÔNG TIN ĐẶC BIỆT THEO LOGIC CHUẨN ---
        if not df_realtime.empty:
            # 1. Khung giờ buồn ngủ nhiều nhất
            drowsy_hour = None
            drowsy_val = None
            if 'drowsiness_detected' in df_realtime.columns:
                df_realtime['hour'] = df_realtime['timestamp'].dt.hour
                drowsy_incr = df_realtime.groupby(['ID', 'hour'])['drowsiness_detected'].apply(
                    lambda x: (x.diff().fillna(0) > 0).sum()).groupby('hour').sum()
                if not drowsy_incr.empty:
                    drowsy_hour = drowsy_incr.idxmax()
                    drowsy_val = int(drowsy_incr.max())

            # 2. Khung giờ chớp mắt ít nhất
            blink_hour = None
            blink_val = None
            if 'blink_count' in df_realtime.columns:
                blink_incr = df_realtime.groupby(df_realtime['timestamp'].dt.hour)['blink_count'].apply(
                    lambda x: x.diff().fillna(0))
                blink_avg = blink_incr.groupby(level=0).apply(lambda x: x[x > 0].mean() if (x > 0).any() else 0)
                blink_avg = blink_avg.replace([float('inf'), float('-inf')], 0).fillna(0)
                if not blink_avg.empty:
                    blink_hour = blink_avg.idxmin()
                    blink_val = round(blink_avg.min(), 2)

            # 3. Khung giờ tư thế thay đổi nhiều nhất
            posture_hour = None
            posture_val = None
            posture_cols = [col for col in ['head_tilt', 'head_side', 'shoulder_angle'] if
                            col in df_realtime.columns]
            if posture_cols:
                std_by_hour = df_realtime.groupby(df_realtime['timestamp'].dt.hour)[posture_cols].std().sum(axis=1)
                if not std_by_hour.empty:
                    posture_hour = std_by_hour.idxmax()
                    posture_val = round(std_by_hour.max(), 2)

            # Create info cards in a 3-column layout
            self._create_info_card(
                self.info_cards_frame,
                f"Khung giờ buồn ngủ nhiều nhất\n{drowsy_hour if drowsy_hour is not None else '-'}h",
                f"Tăng {drowsy_val if drowsy_val is not None else '-'} lần",
                "😴",
                ("#ff9999", "#cc6666"),
                0, 0
            )

            self._create_info_card(
                self.info_cards_frame,
                f"Khung giờ chớp mắt ít nhất\n{blink_hour if blink_hour is not None else '-'}h",
                f"TB {blink_val if blink_val is not None else '-'}",
                "👁️",
                ("#99ccff", "#6699cc"),
                0, 1
            )

            self._create_info_card(
                self.info_cards_frame,
                f"Khung giờ tư thế thay đổi nhiều nhất\n{posture_hour if posture_hour is not None else '-'}h",
                f"Độ biến thiên: {posture_val if posture_val is not None else '-'}",
                "🔄",
                ("#99ff99", "#66cc66"),
                0, 2
            )

            row = 0
        else:
            row = 0

        # NEW: Heatmap for drowsiness events by hour and day
        if not df_summary.empty and 'number_of_drowsiness' in df_summary.columns and 'begin_timestamp' in df_summary.columns:
            chart_frame = ctk.CTkFrame(self.charts_frame, corner_radius=8)
            chart_frame.grid(row=row, column=0, sticky="nsew", padx=10, pady=8)
            chart_frame.grid_rowconfigure(0, weight=1)
            chart_frame.grid_columnconfigure(0, weight=1)

            fig = Figure(figsize=(16, 5), dpi=100)
            ax = fig.add_subplot(111)

            # Prepare data for heatmap
            df_summary['begin_datetime'] = pd.to_datetime(df_summary['begin_timestamp'])
            df_summary['day_of_week'] = df_summary['begin_datetime'].dt.day_name()
            df_summary['hour'] = df_summary['begin_datetime'].dt.hour

            # Tổng hợp number_of_drowsiness theo ngày trong tuần và giờ
            heatmap_data = df_summary.pivot_table(
                values='number_of_drowsiness',
                index='day_of_week',
                columns='hour',
                aggfunc='sum',
                fill_value=0
            )
            # Reindex để đảm bảo tất cả các ngày trong tuần được hiển thị
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(days, fill_value=0)

            # Vẽ biểu đồ nhiệt
            sns.heatmap(heatmap_data, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
                        cbar_kws={'label': 'Số sự kiện buồn ngủ'})
            ax.set_title("Drowsiness Events by Hour and Day", fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel("Giờ trong ngày", fontsize=11)
            ax.set_ylabel("Ngày trong tuần", fontsize=11)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            canvas.draw()
            row += 1

        # NEW: Heatmap for blink per minute by hour and day
        if not df_summary.empty and 'blink_per_minute' in df_summary.columns and 'begin_timestamp' in df_summary.columns:
            chart_frame = ctk.CTkFrame(self.charts_frame, corner_radius=8)
            chart_frame.grid(row=row, column=0, sticky="nsew", padx=10, pady=8)
            chart_frame.grid_rowconfigure(0, weight=1)
            chart_frame.grid_columnconfigure(0, weight=1)

            fig = Figure(figsize=(16, 5), dpi=100)
            ax = fig.add_subplot(111)

            # Prepare data for heatmap
            df_summary['begin_datetime'] = pd.to_datetime(df_summary['begin_timestamp'])
            df_summary['day_of_week'] = df_summary['begin_datetime'].dt.day_name()
            df_summary['hour'] = df_summary['begin_datetime'].dt.hour

            # Tổng hợp blink_per_minute theo ngày trong tuần và giờ
            heatmap_data = df_summary.pivot_table(
                values='blink_per_minute',
                index='day_of_week',
                columns='hour',
                aggfunc='mean',  # Sử dụng mean để lấy trung bình số lần chớp mắt mỗi phút
                fill_value=0
            )
            # Reindex để đảm bảo tất cả các ngày trong tuần được hiển thị
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(days, fill_value=0)

            # Vẽ biểu đồ nhiệt
            sns.heatmap(heatmap_data, ax=ax, cmap="YlGnBu", annot=True, fmt=".1f",
                        cbar_kws={'label': 'Số lần chớp mắt mỗi phút'})
            ax.set_title("Blink per Minute by Hour and Day", fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel("Giờ trong ngày", fontsize=11)
            ax.set_ylabel("Ngày trong tuần", fontsize=11)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            canvas.draw()
            row += 1

        if not df_realtime.empty and {'posture_status', 'time'}.issubset(df_realtime.columns):
            chart_frame = ctk.CTkFrame(self.charts_frame, corner_radius=8)
            chart_frame.grid(row=row, column=0, sticky="nsew", padx=10, pady=8)
            chart_frame.grid_rowconfigure(0, weight=1)
            chart_frame.grid_columnconfigure(0, weight=1)

            fig = Figure(figsize=(16, 4), dpi=100)
            ax = fig.add_subplot(111)

            # Tạo cột giờ từ cột time
            df_realtime['hour'] = pd.to_datetime(df_realtime['time'], format='%H:%M:%S').dt.hour

            # Đếm số lần sai tư thế theo giờ
            bad_by_hour = df_realtime[df_realtime['posture_status'] == 'poor']['hour'].value_counts().sort_index()

            sns.barplot(
                x=bad_by_hour.index,
                y=bad_by_hour.values,
                ax=ax,
                hue=bad_by_hour.index,
                palette='Reds_d',
                legend=False
            )
            ax.set_title("Số lần sai tư thế theo khung giờ")
            ax.set_xlabel("Giờ trong ngày")
            ax.set_ylabel("Số lần (poor)")
            ax.set_xticks(range(0, 24))
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            canvas.draw()
            row += 1

        if not df_summary.empty:
            chart_frame = ctk.CTkFrame(self.charts_frame, corner_radius=8)
            chart_frame.grid(row=row, column=0, sticky="nsew", padx=10, pady=8)
            fig = Figure(figsize=(6, 6), dpi=100)
            ax = fig.add_subplot(111, projection='polar')

            kpi = ['blink_per_minute', 'number_of_drowsiness', 'bad_posture_count', 'avg_distance']
            values = df_summary[kpi].mean().values
            values = (values - values.min()) / (values.max() - values.min() + 1e-9)
            angles = [n / float(len(kpi)) * 2 * 3.14159 for n in range(len(kpi))]
            values = list(values) + [values[0]]
            angles += angles[:1]
            ax.plot(angles, values, 'b-', linewidth=2)
            ax.fill(angles, values, 'b', alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(kpi)
            ax.set_ylim(0, 1)
            ax.set_title("Radar KPI trung bình")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            canvas.draw()
            row += 1

        # Reset scroll position
        self.charts_canvas.yview_moveto(0)