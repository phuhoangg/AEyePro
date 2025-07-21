import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from models.master_agent import MasterAgent
from models.recommend_agent import RecommendAgent
from models.retrieval_agent import RetrievalAgent
from Execution.executor import ExecutorService

class AIAssistantView:
    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.log_lines = []
        self.executor_service = ExecutorService()
        self._setup_ui()
        self._setup_backend()

    def _setup_backend(self):
        # Khởi tạo các agent backend (có thể mở rộng logic kết nối thực tế)
        try:
            self.master_agent = MasterAgent()
            self.recommend_agent = RecommendAgent()
            self.retrieval_agent = RetrievalAgent()
            self.master_agent.set_agents(self.recommend_agent, self.retrieval_agent)
            self._log("Backend agents initialized.")
        except Exception as e:
            self._log(f"Backend init error: {e}")

    def _setup_ui(self):
        self.parent.grid_rowconfigure(1, weight=1)
        self.parent.grid_columnconfigure(0, weight=0)
        self.parent.grid_columnconfigure(1, weight=1)
        self.parent.grid_columnconfigure(2, weight=0)

        # Menu button (hamburger) - đồng bộ với monitoring_view
        self.menu_btn = ctk.CTkButton(self.parent, text="☰", width=60, height=60, corner_radius=12, font=ctk.CTkFont(size=28, weight="bold"))
        self.menu_btn.grid(row=0, column=0, sticky="nw", padx=(10, 0), pady=10)
        self.menu_btn.bind("<Button-1>", self._show_menu)
        self.menu_popup = None

        # Chat frame (bên trái)
        self.chat_frame = ctk.CTkFrame(self.parent, corner_radius=12)
        self.chat_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(80, 10), pady=10)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.chat_frame, text="AI Assistant", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, sticky="ew", pady=(10, 5))

        # Khu vực hội thoại
        self.chat_display = ctk.CTkTextbox(self.chat_frame, height=400, font=ctk.CTkFont(size=12))
        self.chat_display.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_display.insert("end", "Bot: Xin chào! Bạn cần hỏi gì? (VD: Xin chào, Bạn khỏe không?)\n")
        self.chat_display.configure(state="disabled")

        # Input frame
        input_frame = ctk.CTkFrame(self.chat_frame, corner_radius=10)
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)

        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text="Nhập tin nhắn...", font=ctk.CTkFont(size=12))
        self.chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_input.bind("<Return>", lambda event: self.send_message())

        send_button = ctk.CTkButton(input_frame, text="Gửi", width=80, command=self.send_message, corner_radius=8, hover_color="#1E88E5")
        send_button.grid(row=0, column=1)

        # Log/debug frame (bên phải)
        self.log_frame = ctk.CTkFrame(self.parent, width=220, corner_radius=12)
        self.log_frame.grid(row=1, column=2, sticky="nse", padx=(0, 10), pady=10)
        ctk.CTkLabel(self.log_frame, text="Log/Debug", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))
        self.log_box = ctk.CTkTextbox(self.log_frame, width=200, height=400, font=ctk.CTkFont(size=12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_box.insert("end", "Log initialized...\n")
        self.log_box.configure(state="disabled")

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

    def send_message(self):
        message = self.chat_input.get().strip()
        if not message:
            return
        self._append_chat("Bạn", message)
        self.chat_input.delete(0, "end")
        # Gọi backend trả lời (bằng async, không block UI)
        self.generate_ai_response_async(message)

    def _append_chat(self, sender, message):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{sender}: {message}\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{msg}\n")
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def generate_ai_response_async(self, message):
        # Sử dụng ExecutorService để không block UI
        self._log(f"[DEBUG] Đang gọi process_query từ MasterAgent với message: {message}")
        def callback(future):
            try:
                response = future.result()
                self._log(f"[DEBUG] Đã nhận phản hồi từ MasterAgent: {response}")
            except Exception as e:
                response = f"Lỗi khi lấy phản hồi: {e}"
                self._log(f"[DEBUG] Lỗi khi lấy phản hồi từ MasterAgent: {e}")
            self._append_chat("Bot", response)
            self._log(f"Bot: {response}")
        # Đưa process_query vào thread pool (chạy sync qua asyncio.run)
        import asyncio
        def run_query():
            try:
                return asyncio.run(self.master_agent.process_query(message))
            except RuntimeError:
                # Nếu event loop đã chạy, dùng loop mới
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(self.master_agent.process_query(message))
        future = self.executor_service.submit(run_query)
        future.add_done_callback(lambda fut: self.parent.after(0, lambda: callback(fut)))