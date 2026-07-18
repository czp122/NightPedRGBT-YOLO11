from __future__ import annotations

import os
import platform
import queue
import subprocess
import sys
import time
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)


def _restart_with_safe_mkl_environment() -> None:
    """Restart direct source runs before third-party DLLs initialize in IDE launchers."""
    if __name__ != "__main__" or getattr(sys, "frozen", False):
        return
    if os.environ.get("MKL_THREADING_LAYER", "").upper() == "SEQUENTIAL":
        return

    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    print("[BOOT] 正在使用兼容的MKL线程环境重新启动程序", flush=True)
    return_code = subprocess.call(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        env=environment,
    )
    raise SystemExit(return_code)


_restart_with_safe_mkl_environment()

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Conda MKL and PyTorch may otherwise initialize different Intel OpenMP DLLs.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import cv2
import torch
from PIL import Image, ImageTk

BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
OUTPUT_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NightPedRGBT"
    if getattr(sys, "frozen", False)
    else Path(PROJECT_ROOT)
)

from app_utils.acceleration import (
    ComputeDevice,
    available_device_options,
    clear_cuda_cache,
    configure_compute_runtime,
    is_cuda_runtime_error,
    resolve_compute_device,
    synchronize_if_cuda,
)
from app_utils.camera import CameraDevice, choose_auto_camera_setup, discover_cameras
from app_utils.data_loader import DatasetManager
from app_utils.fusion import FusionEngine
from app_utils.preprocess import to_uint8
from app_utils.session import DetectionRecorder, EventClipRecorder, SettingsStore, normalize_settings
from app_utils.system_info import get_cpu_name, get_os_display_name
from app_utils.version import APP_VERSION

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class MainApp:
    COLORS = {
        "bg": "#0b1120",
        "panel": "#111827",
        "panel_alt": "#172033",
        "border": "#263247",
        "text": "#e5edf7",
        "muted": "#8fa1b8",
        "accent": "#2f80ed",
        "accent_hover": "#4b95f5",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "image": "#050914",
    }
    PERFORMANCE_LABELS = {
        "精度": "quality",
        "均衡": "balanced",
        "流畅": "smooth",
    }
    IR_PREPROCESS_MAX_SIDE = {
        "quality": None,
        "balanced": 384,
        "smooth": 256,
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"夜间行人智能检测系统 v{APP_VERSION}")
        screen_width = max(1024, self.root.winfo_screenwidth())
        screen_height = max(720, self.root.winfo_screenheight())
        window_width = min(1500, max(1100, screen_width - 40))
        window_height = min(900, max(650, screen_height - 90))
        window_x = max(0, (screen_width - window_width) // 2)
        window_y = max(0, (screen_height - window_height) // 3)
        self.root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
        self.root.minsize(min(1100, window_width), min(650, window_height))

        self.cpu_threads = configure_compute_runtime()
        self.active_device = resolve_compute_device("auto", self.cpu_threads)
        self.data_manager = None
        self.fusion_engine = FusionEngine()
        self.models = {}
        self.model_channels = {}
        self.current_model = None
        self.curr_img_idx = 0

        self.settings_store = SettingsStore(OUTPUT_ROOT / "runs" / "gui_config" / "settings.json")
        self.settings = self.settings_store.load()
        performance_label = next(
            (
                label
                for label, value in self.PERFORMANCE_LABELS.items()
                if value == self.settings.get("performance_mode")
            ),
            "均衡",
        )

        self.path_var = tk.StringVar(value=self.settings.get("dataset_path", ""))
        self.split_var = tk.StringVar(value="test")
        self.model_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="auto")
        self.imgsz_var = tk.IntVar(value=int(self.settings.get("imgsz", 384)))
        self.conf_var = tk.DoubleVar(value=float(self.settings.get("conf", 0.25)))
        self.iou_var = tk.DoubleVar(value=float(self.settings.get("iou", 0.45)))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.performance_var = tk.StringVar(value=performance_label)
        self.tracking_var = tk.BooleanVar(value=bool(self.settings.get("tracking_enabled", True)))
        self.alert_var = tk.BooleanVar(value=bool(self.settings.get("alert_enabled", True)))
        self.count_var = tk.StringVar(value="当前:0  累计:0  进入警戒区:0")
        self.info_var = tk.StringVar(value="请先选择 LLVIP 路径并加载模型")
        self.run_state_var = tk.StringVar(value="就绪")
        self.source_status_var = tk.StringVar(value="尚未选择输入源")
        self.current_count_metric_var = tk.StringVar(value="0")
        self.unique_count_metric_var = tk.StringVar(value="0")
        self.alert_count_metric_var = tk.StringVar(value="0")
        self.fps_metric_var = tk.StringVar(value="--")
        self.latency_metric_var = tk.StringVar(value="--")
        self.device_metric_var = tk.StringVar(value=self.active_device.label)

        self._lock = threading.Lock()
        self._worker = None
        self._image_pending = False
        self._img_refs = {}
        self._last_view_images = {}
        self._display_geometry = {}
        self._ui_tasks = queue.SimpleQueue()
        self._video_ui_lock = threading.Lock()
        self._video_ui_payload = None
        self._closing = False
        self._fps_ema = 0.0
        self._fps_times = deque(maxlen=30)
        self.video_predict_settings = None
        self.video_model_name = ""
        self.video_performance_mode = "balanced"
        self.video_tracking_enabled = True
        self.video_alert_enabled = True
        self.video_ir_frame_offset = 0
        self.video_ir_shift = (0, 0)
        self._video_frame_index = 0
        self._has_inference_result = False
        self._last_detections = []
        self._seen_track_ids = set()
        self._inside_track_ids = set()
        self._anonymous_inside_count = 0
        self._anonymous_scene_count = 0
        self._anonymous_seen_count = 0
        self._alert_entry_count = 0
        self._alert_flash_until = 0.0
        roi = self.settings.get("alert_roi")
        self.alert_roi = tuple(map(float, roi)) if isinstance(roi, list) and len(roi) == 4 else None
        self._roi_selecting = False
        self._roi_drag_start = None
        self._advanced_expanded = False
        self._log_expanded = False

        self.video_worker = None
        self.video_running = False
        self.video_paused = False
        self.video_source = None
        self.ir_video_source = None
        self.video_backend = None
        self.ir_video_backend = None
        self.stream_mode = "video"
        self.camera_scan_worker = None
        self._camera_scan_generation = 0
        self.cap_vis = None
        self.cap_ir = None
        self.output_writer = None
        self.output_writer_path = None
        self.output_path = None
        self.latest_result_frame = None
        self.latest_result_base_frame = None
        self.save_dir = OUTPUT_ROOT / "runs" / "gui_results"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.detection_recorder = DetectionRecorder()
        self.event_recorder = EventClipRecorder(
            self.save_dir / "events",
            pre_seconds=float(self.settings.get("pre_event_seconds", 3.0)),
            post_seconds=float(self.settings.get("post_event_seconds", 5.0)),
            cooldown_seconds=float(self.settings.get("alarm_cooldown_seconds", 5.0)),
            logger=self.log,
        )

        self._build_ui()
        self.root.after(15, self._drain_ui_tasks)
        self._apply_device_selection(log_selection=True)
        self._load_default_model_if_exist()
        self._set_no_image_all()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def log(self, s: str):
        print(s)
        self._post_ui(self._append_log, str(s))

    def _post_ui(self, callback, *args):
        if self._closing:
            return
        if threading.current_thread() is threading.main_thread():
            callback(*args)
        else:
            self._ui_tasks.put((callback, args))

    def _drain_ui_tasks(self):
        if self._closing:
            return
        try:
            for _ in range(100):
                try:
                    callback, args = self._ui_tasks.get_nowait()
                except queue.Empty:
                    break
                callback(*args)

            with self._video_ui_lock:
                payload = self._video_ui_payload
                self._video_ui_payload = None
            if payload is not None:
                self._update_ui_images(*payload)
        finally:
            if not self._closing:
                self.root.after(15, self._drain_ui_tasks)

    def _append_log(self, s: str):
        self.log_text.config(state="normal")
        upper = s.upper()
        if "[ERROR]" in upper or "失败" in s or "出错" in s:
            tag = "error"
        elif "[WARN]" in upper or "警告" in s:
            tag = "warning"
        elif "[ALERT]" in upper:
            tag = "alert"
        elif "[ACCEL]" in upper or "[CAM]" in upper:
            tag = "accent"
        else:
            tag = "normal"
        self.log_text.insert("end", s + "\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _apply_device_selection(self, event=None, log_selection: bool = True):
        previous = self.active_device
        selected = resolve_compute_device(self.device_var.get(), self.cpu_threads)
        device_changed = previous.predict_arg != selected.predict_arg
        restart_video = device_changed and self.video_running
        if device_changed and not self._wait_for_image_worker():
            self.device_var.set(previous.predict_arg if previous.is_cuda else "cpu")
            return
        if device_changed and self.video_worker and self.video_worker.is_alive() and not self.stop_video():
            self.device_var.set(previous.predict_arg if previous.is_cuda else "cpu")
            return

        self.active_device = selected
        if hasattr(self, "device_metric_var"):
            self.device_metric_var.set(selected.label)
        if device_changed:
            for model in self.models.values():
                model.predictor = None

        if selected.fallback_reason:
            self.log(f"[ACCEL] {selected.fallback_reason}")
        if log_selection:
            self.log(f"[ACCEL] 当前推理设备: {selected.label}")
        if restart_video:
            self.log("[ACCEL] 推理设备已改变，正在重新开始视频")
            self.start_video_inference()

    def _wait_for_image_worker(self, timeout: float | None = 10.0) -> bool:
        worker = self._worker
        if worker is None or not worker.is_alive() or worker is threading.current_thread():
            return True
        worker.join(timeout=timeout)
        if worker.is_alive():
            self.log("[GUI] 图片推理尚未结束，请稍后重试")
            return False
        return True

    def _read_predict_settings(self) -> dict:
        try:
            imgsz = int(self.imgsz_var.get())
            conf = float(self.conf_var.get())
            iou = float(self.iou_var.get())
        except (tk.TclError, TypeError, ValueError) as error:
            raise ValueError("imgsz、conf 和 iou 必须填写有效数字") from error
        if not 32 <= imgsz <= 2048:
            raise ValueError("imgsz 必须在 32 到 2048 之间")
        if not 0.01 <= conf <= 1.0:
            raise ValueError("conf 必须在 0.01 到 1.0 之间")
        if not 0.01 <= iou <= 1.0:
            raise ValueError("iou 必须在 0.01 到 1.0 之间")
        return {"imgsz": imgsz, "conf": conf, "iou": iou}

    def _read_video_speed(self) -> float:
        try:
            speed = float(self.speed_var.get())
        except (tk.TclError, TypeError, ValueError) as error:
            raise ValueError("播放倍速必须填写有效数字") from error
        if not 0.25 <= speed <= 4.0:
            raise ValueError("播放倍速必须在 0.25 到 4.0 之间")
        return speed

    def _performance_key(self) -> str:
        return self.PERFORMANCE_LABELS.get(self.performance_var.get(), "balanced")

    @staticmethod
    def _video_profile_settings(settings: dict, performance_mode: str) -> dict:
        """Make the performance selector affect inference resolution as advertised."""
        effective = dict(settings)
        requested = int(effective["imgsz"])
        if performance_mode == "quality":
            effective["imgsz"] = max(requested, 640)
        elif performance_mode == "smooth":
            effective["imgsz"] = min(requested, 320)
        else:
            effective["imgsz"] = min(max(requested, 384), 640)
        return effective

    def _collect_settings(self) -> dict:
        try:
            predict_settings = self._read_predict_settings()
        except ValueError:
            predict_settings = {
                "imgsz": int(self.settings.get("imgsz", 384)),
                "conf": float(self.settings.get("conf", 0.25)),
                "iou": float(self.settings.get("iou", 0.45)),
            }
        alert_roi = self.alert_roi
        return {
            "dataset_path": self.path_var.get().strip(),
            **predict_settings,
            "performance_mode": self._performance_key(),
            "tracking_enabled": bool(self.tracking_var.get()),
            "alert_enabled": bool(self.alert_var.get()),
            "ir_frame_offset": int(self.settings.get("ir_frame_offset", 0)),
            "ir_shift_x": int(self.settings.get("ir_shift_x", 0)),
            "ir_shift_y": int(self.settings.get("ir_shift_y", 0)),
            "pre_event_seconds": float(self.settings.get("pre_event_seconds", 3.0)),
            "post_event_seconds": float(self.settings.get("post_event_seconds", 5.0)),
            "alarm_cooldown_seconds": float(self.settings.get("alarm_cooldown_seconds", 5.0)),
            "alert_roi": list(alert_roi) if alert_roi is not None else None,
        }

    def _save_settings(self):
        self.settings = normalize_settings(self._collect_settings())
        try:
            self.settings_store.save(self.settings)
        except OSError as error:
            self.log(f"[CONFIG] 保存设置失败: {error}")

    def _reset_tracking_state(self):
        self._video_frame_index = 0
        self._has_inference_result = False
        self._last_detections = []
        self._seen_track_ids.clear()
        self._inside_track_ids.clear()
        self._anonymous_inside_count = 0
        self._anonymous_scene_count = 0
        self._anonymous_seen_count = 0
        self._alert_entry_count = 0
        self._post_ui(self._set_count_metrics, 0, 0, 0)
        predictor = getattr(self.current_model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            try:
                tracker.reset()
            except Exception:
                pass

    def toggle_roi_selection(self):
        self._roi_selecting = not self._roi_selecting
        self._roi_drag_start = None
        cursor = "crosshair" if self._roi_selecting else ""
        self.lbl_result.inner_label.config(cursor=cursor)
        self.info_var.set("请在检测结果画面中拖动鼠标设置警戒区" if self._roi_selecting else "已取消警戒区设置")

    def clear_alert_roi(self):
        self.alert_roi = None
        self._inside_track_ids = set()
        self._save_settings()
        self._refresh_result_overlay()
        self.log("[ALERT] 已清除警戒区域")

    def _refresh_result_overlay(self):
        if self.latest_result_base_frame is None:
            return
        preview = self.latest_result_base_frame.copy()
        self._draw_alert_roi(preview)
        self.latest_result_frame = preview.copy()
        self._show_image(preview, self.lbl_result.inner_label)

    def _event_to_normalized(self, event, *, clamp: bool = False) -> tuple[float, float] | None:
        geometry = self._display_geometry.get(self.lbl_result.inner_label)
        if geometry is None:
            return None
        _orig_w, _orig_h, render_w, render_h, offset_x, offset_y = geometry
        if render_w <= 0 or render_h <= 0:
            return None
        raw_x = (event.x - offset_x) / render_w
        raw_y = (event.y - offset_y) / render_h
        if not clamp and not (0.0 <= raw_x <= 1.0 and 0.0 <= raw_y <= 1.0):
            return None
        nx = min(1.0, max(0.0, raw_x))
        ny = min(1.0, max(0.0, raw_y))
        return nx, ny

    def _on_roi_press(self, event):
        if not self._roi_selecting:
            return
        self._roi_drag_start = self._event_to_normalized(event)

    def _on_roi_drag(self, event):
        if not self._roi_selecting or self._roi_drag_start is None:
            return
        point = self._event_to_normalized(event, clamp=True)
        if point is None:
            return
        x1, y1 = self._roi_drag_start
        x2, y2 = point
        self.alert_roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self._refresh_result_overlay()

    def _on_roi_release(self, event):
        if not self._roi_selecting or self._roi_drag_start is None:
            return
        self._on_roi_drag(event)
        self._roi_drag_start = None
        self._roi_selecting = False
        self.lbl_result.inner_label.config(cursor="")
        if self.alert_roi is None or (
            self.alert_roi[2] - self.alert_roi[0] < 0.02
            or self.alert_roi[3] - self.alert_roi[1] < 0.02
        ):
            self.alert_roi = None
            self.info_var.set("警戒区域过小，请重新设置")
            return
        self._save_settings()
        self.log(f"[ALERT] 警戒区域已设置: {tuple(round(v, 3) for v in self.alert_roi)}")

    def open_sync_settings(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("双模态同步与报警设置")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        values = {
            "ir_frame_offset": tk.IntVar(value=int(self.settings.get("ir_frame_offset", 0))),
            "ir_shift_x": tk.IntVar(value=int(self.settings.get("ir_shift_x", 0))),
            "ir_shift_y": tk.IntVar(value=int(self.settings.get("ir_shift_y", 0))),
            "pre_event_seconds": tk.DoubleVar(value=float(self.settings.get("pre_event_seconds", 3.0))),
            "post_event_seconds": tk.DoubleVar(value=float(self.settings.get("post_event_seconds", 5.0))),
            "alarm_cooldown_seconds": tk.DoubleVar(
                value=float(self.settings.get("alarm_cooldown_seconds", 5.0))
            ),
        }
        fields = [
            ("IR帧偏移", "ir_frame_offset", -60, 60, 1),
            ("IR水平偏移(px)", "ir_shift_x", -100, 100, 1),
            ("IR垂直偏移(px)", "ir_shift_y", -100, 100, 1),
            ("报警前录像(s)", "pre_event_seconds", 0, 10, 0.5),
            ("报警后录像(s)", "post_event_seconds", 1, 30, 0.5),
            ("报警冷却(s)", "alarm_cooldown_seconds", 1, 60, 1),
        ]
        for row, (label, key, minimum, maximum, increment) in enumerate(fields):
            tk.Label(dialog, text=label, anchor="e", width=18).grid(row=row, column=0, padx=8, pady=5)
            tk.Spinbox(
                dialog,
                from_=minimum,
                to=maximum,
                increment=increment,
                textvariable=values[key],
                width=10,
            ).grid(row=row, column=1, padx=8, pady=5)

        def apply_settings():
            try:
                parsed = {key: variable.get() for key, variable in values.items()}
            except (tk.TclError, ValueError) as error:
                messagebox.showerror("设置错误", f"请填写有效的同步与报警参数：\n{error}", parent=dialog)
                return
            restart_video = self.video_running
            if self.video_worker and self.video_worker.is_alive() and not self.stop_video():
                messagebox.showwarning("请稍候", "当前推理仍在结束，请稍后再次保存。", parent=dialog)
                return
            self.settings.update(parsed)
            self._save_settings()
            self.event_recorder.configure(
                self.settings["pre_event_seconds"],
                self.settings["post_event_seconds"],
                self.settings["alarm_cooldown_seconds"],
            )
            self.log(
                "[SYNC] 设置已保存: "
                f"IR帧偏移={self.settings['ir_frame_offset']} "
                f"平移=({self.settings['ir_shift_x']},{self.settings['ir_shift_y']})"
            )
            dialog.destroy()
            if restart_video:
                self.start_video_inference()

        tk.Button(dialog, text="保存", command=apply_settings, width=10).grid(
            row=len(fields), column=0, columnspan=2, pady=10
        )
        dialog.grab_set()

    def export_detection_records(self):
        if len(self.detection_recorder) == 0:
            messagebox.showwarning("提示", "当前没有可导出的检测记录")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="导出检测记录",
            initialdir=str(self.save_dir),
            initialfile=f"detections_{timestamp}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not path:
            return
        output = self.detection_recorder.export(path)
        self.log(f"[RECORD] 已导出 {len(self.detection_recorder)} 条记录: {output}")

    def show_device_diagnostics(self):
        cuda_available = torch.cuda.is_available()
        lines = [
            f"软件版本: v{APP_VERSION}",
            f"系统: {get_os_display_name()}",
            f"Python: {platform.python_version()}",
            f"CPU: {get_cpu_name()}",
            f"CPU逻辑核心: {os.cpu_count() or 1}",
            f"PyTorch: {torch.__version__}",
            f"OpenCV: {cv2.__version__}",
            f"CUDA构建版本: {torch.version.cuda or 'None'}",
            f"CUDA可用: {cuda_available}",
            f"当前设备: {self.active_device.label}",
            f"当前模型: {self.model_var.get() or '未加载'}",
            f"模型通道: {self._current_model_channels() if self.current_model else '-'}",
            f"性能模式: {self.performance_var.get()}",
            f"检测记录: {len(self.detection_recorder)} 条",
        ]
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(index) / (1024**2)
                total = torch.cuda.get_device_properties(index).total_memory / (1024**2)
                lines.append(
                    f"GPU {index}: {torch.cuda.get_device_name(index)} | 显存 {allocated:.0f}/{total:.0f} MB"
                )
        dialog = tk.Toplevel(self.root)
        dialog.title("运行环境诊断")
        dialog.geometry("680x420")
        text_widget = tk.Text(dialog, font=("Consolas", 11), padx=12, pady=12)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert("1.0", "\n".join(lines))
        text_widget.config(state="disabled")

    def _on_runtime_option_change(self, event=None):
        restart_video = self.video_running
        if self.video_worker and self.video_worker.is_alive() and not self.stop_video():
            return
        self._save_settings()
        if restart_video:
            self.log("[GUI] 运行选项已改变，正在重新开始视频")
            self.start_video_inference()

    def _apply_predict_settings(self, event=None):
        try:
            self._read_predict_settings()
        except ValueError as error:
            messagebox.showerror("参数错误", str(error))
            return
        self._on_runtime_option_change()

    def _apply_video_speed(self, event=None):
        try:
            self._read_video_speed()
        except ValueError as error:
            messagebox.showerror("参数错误", str(error))
            return
        self._on_runtime_option_change()

    def _build_ui(self):
        c = self.COLORS
        self.root.configure(bg=c["bg"])
        self._configure_styles()

        header = tk.Frame(self.root, bg=c["panel"], padx=12, pady=9)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            header,
            text="夜间行人智能检测",
            bg=c["panel"],
            fg=c["text"],
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text=f"YOLO11 · RGBT  |  v{APP_VERSION}",
            bg=c["panel"],
            fg=c["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(side=tk.LEFT, padx=(12, 0), pady=(5, 0))
        self.run_state_badge = tk.Label(
            header,
            textvariable=self.run_state_var,
            bg=c["success"],
            fg="#ffffff",
            padx=14,
            pady=4,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self.run_state_badge.pack(side=tk.RIGHT)

        self.toolbar_container = tk.Frame(self.root, bg=c["panel_alt"], padx=10, pady=7)
        self.toolbar_container.pack(side=tk.TOP, fill=tk.X)
        top = tk.Frame(self.toolbar_container, bg=c["panel_alt"])
        top.pack(fill=tk.X)

        tk.Label(top, text="模型", bg=c["panel_alt"], fg=c["muted"]).pack(side=tk.LEFT, padx=(2, 5))
        self.model_combo = ttk.Combobox(
            top,
            textvariable=self.model_var,
            values=[],
            width=25,
            state="readonly",
        )
        self.model_combo.pack(side=tk.LEFT, padx=4)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        ttk.Button(top, text="加载模型", command=self.load_custom_model, style="Secondary.TButton").pack(
            side=tk.LEFT, padx=(2, 10)
        )

        tk.Label(top, text="设备", bg=c["panel_alt"], fg=c["muted"]).pack(side=tk.LEFT, padx=(2, 5))
        self.device_combo = ttk.Combobox(
            top,
            textvariable=self.device_var,
            values=available_device_options(),
            width=7,
            state="readonly",
        )
        self.device_combo.pack(side=tk.LEFT, padx=4)
        self.device_combo.bind("<<ComboboxSelected>>", self._apply_device_selection)

        tk.Label(top, text="性能", bg=c["panel_alt"], fg=c["muted"]).pack(side=tk.LEFT, padx=(8, 5))
        self.performance_combo = ttk.Combobox(
            top,
            textvariable=self.performance_var,
            values=list(self.PERFORMANCE_LABELS),
            width=6,
            state="readonly",
        )
        self.performance_combo.pack(side=tk.LEFT, padx=4)
        self.performance_combo.bind("<<ComboboxSelected>>", self._on_runtime_option_change)
        ttk.Checkbutton(
            top,
            text="跟踪",
            variable=self.tracking_var,
            command=self._on_runtime_option_change,
            style="Toolbar.TCheckbutton",
        ).pack(side=tk.LEFT, padx=(9, 3))
        ttk.Checkbutton(
            top,
            text="警戒",
            variable=self.alert_var,
            command=self._on_runtime_option_change,
            style="Toolbar.TCheckbutton",
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            top,
            text="高级参数 ▾",
            command=self._toggle_advanced_controls,
            style="Ghost.TButton",
        ).pack(side=tk.LEFT, padx=(8, 3))
        self.advanced_toggle_button = top.winfo_children()[-1]

        self.stop_button = ttk.Button(
            top,
            text="停止",
            command=self.stop_video,
            style="Danger.TButton",
            width=8,
        )
        self.stop_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.start_button = ttk.Button(
            top,
            text="▶ 开始检测",
            command=self.start_video_inference,
            style="Primary.TButton",
            width=13,
            state=tk.DISABLED,
        )
        self.start_button.pack(side=tk.RIGHT, padx=5)

        self.advanced_frame = tk.Frame(self.toolbar_container, bg=c["panel_alt"])
        tk.Label(
            self.advanced_frame,
            text="高级推理参数",
            bg=c["panel_alt"],
            fg=c["muted"],
        ).pack(side=tk.LEFT, padx=(2, 10))
        for text, var in (("输入尺寸", self.imgsz_var), ("置信度", self.conf_var), ("IOU", self.iou_var)):
            tk.Label(self.advanced_frame, text=text, bg=c["panel_alt"], fg=c["text"]).pack(
                side=tk.LEFT, padx=(7, 3)
            )
            entry = tk.Entry(
                self.advanced_frame,
                textvariable=var,
                width=7,
                bg=c["bg"],
                fg=c["text"],
                insertbackground=c["text"],
                relief=tk.FLAT,
            )
            entry.pack(side=tk.LEFT, ipady=4)
            entry.bind("<Return>", self._apply_predict_settings)
        tk.Label(
            self.advanced_frame,
            text="修改后按 Enter 应用",
            bg=c["panel_alt"],
            fg=c["muted"],
        ).pack(side=tk.LEFT, padx=12)

        controls = ttk.Notebook(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(7, 0))
        realtime_tab = tk.Frame(controls, bg=c["panel"], padx=8, pady=7)
        evaluation_tab = tk.Frame(controls, bg=c["panel"], padx=8, pady=9)
        controls.add(realtime_tab, text="实时检测")
        controls.add(evaluation_tab, text="数据集评估")

        source_group = self._control_group(realtime_tab, "输入源")
        source_group.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        for index, (text, command) in enumerate(
            (("RGB视频", self.open_visible_video), ("红外视频", self.open_ir_video),
             ("自动摄像头", self.open_auto_camera), ("手动双摄", self.open_rgbt_cameras))
        ):
            ttk.Button(source_group, text=text, command=command, style="Secondary.TButton").grid(
                row=index // 2, column=index % 2, padx=3, pady=3, sticky="ew"
            )

        playback_group = self._control_group(realtime_tab, "运行控制")
        playback_group.pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(
            playback_group,
            text="暂停 / 继续",
            command=self.toggle_pause_video,
            style="Secondary.TButton",
        ).grid(row=0, column=0, padx=3, pady=3)
        tk.Label(playback_group, text="倍速", bg=c["panel_alt"], fg=c["muted"]).grid(
            row=0, column=1, padx=(8, 3)
        )
        speed_spinbox = tk.Spinbox(
            playback_group,
            from_=0.25,
            to=4.0,
            increment=0.25,
            textvariable=self.speed_var,
            width=5,
            command=self._apply_video_speed,
            bg=c["bg"],
            fg=c["text"],
            buttonbackground=c["panel_alt"],
            insertbackground=c["text"],
            relief=tk.FLAT,
        )
        speed_spinbox.grid(row=0, column=2, padx=3, ipady=3)
        speed_spinbox.bind("<Return>", self._apply_video_speed)
        tk.Label(
            playback_group,
            textvariable=self.source_status_var,
            bg=c["panel_alt"],
            fg=c["muted"],
            anchor="w",
        ).grid(row=1, column=0, columnspan=3, padx=3, pady=(3, 0), sticky="w")

        tools_group = self._control_group(realtime_tab, "警戒与输出")
        tools_group.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tool_items = (
            ("设置警戒区", self.toggle_roi_selection),
            ("清除警戒区", self.clear_alert_roi),
            ("同步校准", self.open_sync_settings),
            ("保存结果", self.save_current_result),
            ("导出记录", self.export_detection_records),
            ("设备信息", self.show_device_diagnostics),
        )
        for index, (text, command) in enumerate(tool_items):
            ttk.Button(tools_group, text=text, command=command, style="Secondary.TButton").grid(
                row=index // 2, column=index % 2, padx=3, pady=3, sticky="ew"
            )
            tools_group.grid_columnconfigure(index % 2, weight=1)

        tk.Label(evaluation_tab, text="LLVIP路径", bg=c["panel"], fg=c["muted"]).grid(
            row=0, column=0, padx=(3, 6)
        )
        self.path_entry = tk.Entry(
            evaluation_tab,
            textvariable=self.path_var,
            bg=c["bg"],
            fg=c["text"],
            insertbackground=c["text"],
            relief=tk.FLAT,
        )
        self.path_entry.grid(row=0, column=1, sticky="ew", padx=4, ipady=5)
        evaluation_tab.grid_columnconfigure(1, weight=1)
        ttk.Button(evaluation_tab, text="浏览", command=self.browse_dataset, style="Secondary.TButton").grid(
            row=0, column=2, padx=3
        )
        ttk.Button(
            evaluation_tab,
            text="加载 / 刷新",
            command=self.load_dataset_from_entry,
            style="Primary.TButton",
        ).grid(row=0, column=3, padx=3)
        tk.Label(evaluation_tab, text="子集", bg=c["panel"], fg=c["muted"]).grid(
            row=0, column=4, padx=(10, 3)
        )
        self.split_combo = ttk.Combobox(
            evaluation_tab,
            textvariable=self.split_var,
            values=["train", "test", "val"],
            width=7,
            state="readonly",
        )
        self.split_combo.grid(row=0, column=5, padx=3)
        self.split_combo.bind("<<ComboboxSelected>>", self.on_split_change)
        ttk.Button(evaluation_tab, text="上一张", command=self.prev_img, style="Secondary.TButton").grid(
            row=0, column=6, padx=(10, 3)
        )
        ttk.Button(evaluation_tab, text="下一张", command=self.next_img, style="Secondary.TButton").grid(
            row=0, column=7, padx=3
        )

        logf = tk.Frame(self.root, bg=c["panel"], highlightthickness=1, highlightbackground=c["border"])
        self.log_panel = logf
        log_header = tk.Frame(logf, bg=c["panel"])
        log_header.pack(fill=tk.X)
        tk.Label(
            log_header,
            text="运行日志",
            bg=c["panel"],
            fg=c["text"],
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=10, pady=5)
        log_close_button = ttk.Button(
            log_header,
            text="收起",
            command=self._toggle_log_panel,
            style="Ghost.TButton",
            width=7,
        )
        log_close_button.pack(side=tk.RIGHT, padx=5, pady=2)
        self.log_body = tk.Frame(logf, bg=c["panel"])
        self.log_text = tk.Text(
            self.log_body,
            height=6,
            bg="#080d18",
            fg="#cbd5e1",
            insertbackground=c["text"],
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.log_body.pack(fill=tk.X)
        self.log_text.pack(fill=tk.X, expand=True, padx=8, pady=(0, 8))
        self.log_text.insert("end", "GUI Ready.\n")
        self.log_text.tag_configure("normal", foreground="#cbd5e1")
        self.log_text.tag_configure("accent", foreground="#67e8f9")
        self.log_text.tag_configure("warning", foreground="#fbbf24")
        self.log_text.tag_configure("alert", foreground="#fb923c")
        self.log_text.tag_configure("error", foreground="#f87171")
        self.log_text.config(state="disabled")

        status_bar = tk.Frame(self.root, bg=c["panel_alt"], padx=10, pady=5)
        self.status_bar = status_bar
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))
        tk.Label(
            status_bar,
            textvariable=self.info_var,
            bg=c["panel_alt"],
            fg=c["muted"],
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.log_toggle_button = ttk.Button(
            status_bar,
            text="运行日志",
            command=self._toggle_log_panel,
            style="Ghost.TButton",
            width=9,
        )
        self.log_toggle_button.pack(side=tk.RIGHT, padx=(8, 0))

        metrics = tk.Frame(self.root, bg=c["bg"])
        metrics.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))
        metric_specs = (
            ("当前人数", self.current_count_metric_var, c["accent"]),
            ("累计人数", self.unique_count_metric_var, "#8b5cf6"),
            ("警戒进入", self.alert_count_metric_var, c["danger"]),
            ("实时 FPS", self.fps_metric_var, c["success"]),
            ("推理耗时", self.latency_metric_var, c["warning"]),
            ("计算设备", self.device_metric_var, "#06b6d4"),
        )
        for index, spec in enumerate(metric_specs):
            card = self._metric_card(metrics, *spec)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 3, 0))
            metrics.grid_columnconfigure(index, weight=1, uniform="metric")

        self.main_pane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=c["bg"],
            sashwidth=6,
            sashrelief=tk.FLAT,
            bd=0,
        )
        self.main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)
        result_area = tk.Frame(self.main_pane, bg=c["bg"])
        preview_area = tk.Frame(self.main_pane, bg=c["bg"])
        self.main_pane.add(result_area, stretch="always", minsize=600)
        self.main_pane.add(preview_area, stretch="never", minsize=280)

        self.lbl_result = self._box(result_area, "检测结果  ·  双击放大")
        self.lbl_result.pack(fill=tk.BOTH, expand=True)
        self.lbl_rgb = self._box(preview_area, "可见光 RGB")
        self.lbl_ir = self._box(preview_area, "红外 IR")
        self.lbl_fusion = self._box(preview_area, "融合预览")
        for frame in (self.lbl_rgb, self.lbl_ir, self.lbl_fusion):
            frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.lbl_result.inner_label.bind("<ButtonPress-1>", self._on_roi_press)
        self.lbl_result.inner_label.bind("<B1-Motion>", self._on_roi_drag)
        self.lbl_result.inner_label.bind("<ButtonRelease-1>", self._on_roi_release)
        self.root.after(120, self._set_initial_pane_position)

    def _box(self, parent, title):
        c = self.COLORS
        f = tk.Frame(
            parent,
            bg=c["image"],
            highlightthickness=1,
            highlightbackground=c["border"],
        )
        title_label = tk.Label(
            f,
            text=title,
            bg=c["panel_alt"],
            fg=c["text"],
            anchor="w",
            padx=10,
            pady=5,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        title_label.pack(side=tk.TOP, fill=tk.X)
        l = tk.Label(f, text="暂无画面", bg=c["image"], fg=c["muted"])
        l.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        l.bind("<Double-Button-1>", lambda _event, widget=l, name=title: self._open_image_preview(widget, name))
        f.title_label = title_label
        f.inner_label = l
        return f

    def _configure_styles(self):
        c = self.COLORS
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            ".",
            background=c["panel"],
            foreground=c["text"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "TCombobox",
            fieldbackground=c["bg"],
            background=c["panel_alt"],
            foreground=c["text"],
            arrowcolor=c["text"],
            bordercolor=c["border"],
            lightcolor=c["border"],
            darkcolor=c["border"],
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", c["bg"])],
            foreground=[("readonly", c["text"])],
            selectbackground=[("readonly", c["bg"])],
            selectforeground=[("readonly", c["text"])],
        )
        style.configure("TNotebook", background=c["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=c["panel_alt"],
            foreground=c["muted"],
            padding=(18, 7),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", c["accent"])],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Primary.TButton",
            background=c["accent"],
            foreground="#ffffff",
            borderwidth=0,
            padding=(10, 6),
        )
        style.map(
            "Primary.TButton",
            background=[("active", c["accent_hover"]), ("disabled", "#344055")],
            foreground=[("disabled", "#7b8798")],
        )
        style.configure(
            "Secondary.TButton",
            background=c["panel_alt"],
            foreground=c["text"],
            bordercolor=c["border"],
            padding=(9, 5),
        )
        style.map("Secondary.TButton", background=[("active", "#22304a")])
        style.configure(
            "Danger.TButton",
            background="#7f1d1d",
            foreground="#ffffff",
            borderwidth=0,
            padding=(9, 6),
        )
        style.map("Danger.TButton", background=[("active", c["danger"])])
        style.configure(
            "Ghost.TButton",
            background=c["panel_alt"],
            foreground=c["muted"],
            borderwidth=0,
            padding=(7, 4),
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#22304a")],
            foreground=[("active", c["text"])],
        )
        style.configure(
            "Toolbar.TCheckbutton",
            background=c["panel_alt"],
            foreground=c["text"],
            indicatorcolor=c["bg"],
        )
        style.map(
            "Toolbar.TCheckbutton",
            background=[("active", c["panel_alt"])],
            indicatorcolor=[("selected", c["accent"])],
        )
        self.root.option_add("*TCombobox*Listbox.background", c["bg"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])

    def _control_group(self, parent, title):
        c = self.COLORS
        return tk.LabelFrame(
            parent,
            text=title,
            bg=c["panel_alt"],
            fg=c["muted"],
            bd=0,
            padx=6,
            pady=4,
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    def _metric_card(self, parent, title, variable, accent):
        c = self.COLORS
        card = tk.Frame(
            parent,
            bg=c["panel"],
            highlightthickness=1,
            highlightbackground=c["border"],
            padx=10,
            pady=7,
        )
        tk.Frame(card, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 9))
        text = tk.Frame(card, bg=c["panel"])
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            text,
            text=title,
            bg=c["panel"],
            fg=c["muted"],
            anchor="w",
            font=("Microsoft YaHei UI", 8),
        ).pack(fill=tk.X)
        tk.Label(
            text,
            textvariable=variable,
            bg=c["panel"],
            fg=c["text"],
            anchor="w",
            font=("Consolas", 12, "bold"),
        ).pack(fill=tk.X)
        return card

    def _toggle_advanced_controls(self):
        self._advanced_expanded = not self._advanced_expanded
        if self._advanced_expanded:
            self.advanced_frame.pack(fill=tk.X, pady=(7, 0))
            self.advanced_toggle_button.config(text="高级参数 ▴")
        else:
            self.advanced_frame.pack_forget()
            self.advanced_toggle_button.config(text="高级参数 ▾")

    def _toggle_log_panel(self):
        self._log_expanded = not self._log_expanded
        if self._log_expanded:
            self.log_panel.pack(
                side=tk.BOTTOM,
                fill=tk.X,
                padx=10,
                pady=(0, 6),
                after=self.status_bar,
            )
            self.log_toggle_button.config(text="收起日志")
        else:
            self.log_panel.pack_forget()
            self.log_toggle_button.config(text="运行日志")

    def _set_initial_pane_position(self):
        try:
            self.main_pane.sash_place(0, max(650, int(self.main_pane.winfo_width() * 0.72)), 0)
        except tk.TclError:
            pass

    def _set_run_state(self, text: str, level: str = "idle"):
        variable = getattr(self, "run_state_var", None)
        if variable is not None:
            variable.set(text)
        badge = getattr(self, "run_state_badge", None)
        if badge is None:
            return
        color = {
            "running": self.COLORS["success"],
            "paused": self.COLORS["warning"],
            "warning": self.COLORS["warning"],
            "error": self.COLORS["danger"],
        }.get(level, self.COLORS["accent"])
        badge.config(bg=color)

    def _set_count_metrics(self, current: int, unique: int, alerts: int):
        self.count_var.set(f"当前:{current}  累计:{unique}  进入警戒区:{alerts}")
        for attribute, value in (
            ("current_count_metric_var", current),
            ("unique_count_metric_var", unique),
            ("alert_count_metric_var", alerts),
        ):
            variable = getattr(self, attribute, None)
            if variable is not None:
                variable.set(str(value))

    def _refresh_source_readiness(self):
        has_model = getattr(self, "current_model", None) is not None
        channels = self._current_model_channels() if has_model else 0
        has_rgb = getattr(self, "video_source", None) is not None
        needs_ir = channels >= 6
        has_ir = getattr(self, "ir_video_source", None) is not None
        valid_pair = not needs_ir or (has_ir and self.video_source != self.ir_video_source)
        ready = has_model and has_rgb and valid_pair

        parts = [f"模型 {channels}ch" if has_model else "未加载模型"]
        parts.append("RGB ✓" if has_rgb else "RGB 未选择")
        if needs_ir:
            parts.append("IR ✓" if has_ir else "IR 未选择")
        source_variable = getattr(self, "source_status_var", None)
        if source_variable is not None:
            source_variable.set("  ·  ".join(parts))
        button = getattr(self, "start_button", None)
        if button is not None:
            button.config(state=tk.NORMAL if ready and not getattr(self, "video_running", False) else tk.DISABLED)

    def _on_video_stopped(self, error: str | None = None):
        if error:
            self._set_run_state("运行异常", "error")
        else:
            self._set_run_state("已停止", "idle")
        self._refresh_source_readiness()

    def _open_image_preview(self, image_label, title):
        cv_img = self._last_view_images.get(image_label)
        if cv_img is None:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=self.COLORS["image"])
        dialog.geometry("1200x760")
        try:
            dialog.state("zoomed")
        except tk.TclError:
            pass
        preview_label = tk.Label(dialog, bg=self.COLORS["image"])
        preview_label.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            dialog,
            text="双击或按 Esc 关闭",
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            pady=5,
        ).pack(side=tk.BOTTOM, fill=tk.X)

        def render(_event=None):
            image = to_uint8(cv_img)
            height, width = image.shape[:2]
            target_w = max(320, preview_label.winfo_width() - 10)
            target_h = max(240, preview_label.winfo_height() - 10)
            scale = min(target_w / max(width, 1), target_h / max(height, 1))
            resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB) if resized.ndim == 2 else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            preview_label.config(image=photo)
            preview_label.image = photo

        preview_label.bind("<Configure>", render)
        preview_label.bind("<Double-Button-1>", lambda _event: dialog.destroy())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.after(80, render)

    def _set_stream_titles(self, use_ir: bool):
        self.lbl_rgb.title_label.config(text="可见光 RGB")
        self.lbl_ir.title_label.config(text="红外 IR" if use_ir else "红外 IR（未使用）")
        self.lbl_fusion.title_label.config(text="融合预览" if use_ir else "RGB 预览")
        self.lbl_result.title_label.config(text="检测结果  ·  红框=预测  ·  双击放大")

    def browse_dataset(self):
        p = filedialog.askdirectory(title="请选择 LLVIP 根目录(包含 visible/infrared)")
        if p:
            self.path_var.set(p)

    def load_dataset_from_entry(self):
        if not self._wait_for_image_worker():
            return
        try:
            self.data_manager = DatasetManager(self.path_var.get().strip(), logger=self.log)
            self.on_split_change(None)
        except Exception as e:
            self.data_manager = None
            self.log(f"[GUI] 数据集加载失败: {e}")
            messagebox.showerror("路径错误", str(e))

    def on_split_change(self, event):
        if self.data_manager is None:
            return
        if not self._wait_for_image_worker():
            return
        split = self.split_combo.get().strip() or self.split_var.get().strip()
        n = self.data_manager.load_split(split)
        self.curr_img_idx = 0
        if n <= 0:
            self.info_var.set(f"加载子集: {split}, 0 张")
            self._set_no_image_all()
            return
        self.info_var.set(f"加载子集: {split}, 共 {n} 张图片")
        self.process_current_image_async()

    def _infer_model_channels(self, model) -> int:
        try:
            first_layer = model.model.model[0]
            conv = getattr(first_layer, "conv", None)
            if conv is not None and hasattr(conv, "in_channels"):
                return int(conv.in_channels)
            if hasattr(first_layer, "in_channels"):
                return int(first_layer.in_channels)
        except Exception:
            pass
        return 3

    def _register_model(self, path: str):
        abs_path = os.path.abspath(path)

        for name, model in self.models.items():
            old_path = getattr(model, "_source_path", None)
            if old_path and os.path.abspath(old_path) == abs_path:
                self.model_var.set(name)
                self.current_model = model
                self.model_combo["values"] = list(self.models.keys())
                self.log(f"[GUI] 模型已存在，切换到: {name}")
                self._refresh_source_readiness()
                return

        model = YOLO(path)
        model._source_path = abs_path

        base_name = os.path.basename(path)
        name = base_name
        idx = 2

        while name in self.models:
            stem, ext = os.path.splitext(base_name)
            name = f"{stem}_{idx}{ext}"
            idx += 1

        ch = self._infer_model_channels(model)
        self.models[name] = model
        self.model_channels[name] = ch
        self.model_combo["values"] = list(self.models.keys())
        self.model_var.set(name)
        self.current_model = model

        mode = "RGB/IR 6通道" if ch == 6 else "标准3通道"
        self.log(f"[GUI] 模型已加载: {name} ({mode})")
        self._refresh_source_readiness()

    def _current_model_channels(self) -> int:
        name = self.model_var.get()
        return int(self.model_channels.get(name, 3))

    def _select_loaded_model_by_channels(self, channels: int) -> bool:
        for name, model in self.models.items():
            if int(self.model_channels.get(name, 3)) == channels:
                self.model_var.set(name)
                self.current_model = model
                self.log(f"[GUI] 已自动切换到{channels}通道模型: {name}")
                self._refresh_source_readiness()
                return True
        return False

    def _ensure_model_channels(self, channels: int) -> bool:
        if self.current_model is not None and self._current_model_channels() == channels:
            return True
        if self._select_loaded_model_by_channels(channels):
            return True

        if channels == 3:
            candidates = [BUNDLE_ROOT / "yolo11n.pt"]
        else:
            runs_dir = Path(PROJECT_ROOT) / "runs"
            bundled_rgbt = BUNDLE_ROOT / "models" / "rgbt_best.pt"
            candidates = [bundled_rgbt] if bundled_rgbt.exists() else []
            if runs_dir.exists():
                run_candidates = list(runs_dir.rglob("best.pt"))
                run_candidates.sort(
                    key=lambda p: (
                        int("rgbt" in str(p).lower()),
                        int("cbam" in str(p).lower()),
                        p.stat().st_mtime,
                    ),
                    reverse=True,
                )
                candidates.extend(run_candidates)

        for path in candidates:
            if not path.exists():
                continue
            try:
                self._register_model(str(path))
            except Exception as e:
                self.log(f"[GUI] 自动加载{channels}通道模型失败: {e}")
                continue
            if self._current_model_channels() == channels:
                return True
        return False

    def _model_input_with_preprocessed_ir(self, rgb, ir_preprocessed):
        if self.current_model is None:
            return None
        ch = self._current_model_channels()
        if ch >= 6:
            return self.fusion_engine.for_model_with_preprocessed_ir(rgb, ir_preprocessed)
        return rgb

    def _load_default_model_if_exist(self):
        if YOLO is None:
            return

        candidates = []
        runs_dir = Path(PROJECT_ROOT) / "runs"
        bundled_rgbt = BUNDLE_ROOT / "models" / "rgbt_best.pt"
        if bundled_rgbt.exists():
            candidates.append(bundled_rgbt)
        if runs_dir.exists():
            best_weights = list(runs_dir.rglob("best.pt"))

            def weight_score(p: Path):
                s = str(p).lower()
                return (
                    int("rgbt" in s),
                    int("cbam" in s),
                    p.stat().st_mtime,
                )

            candidates.extend(sorted(best_weights, key=weight_score, reverse=True))

        candidates.extend([
            BUNDLE_ROOT / "best.pt",
            BUNDLE_ROOT / "yolo11n.pt",
        ])

        seen = set()
        for p in candidates:
            p = Path(p)
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.exists():
                try:
                    self._register_model(str(p))
                    return
                except Exception as e:
                    self.log(f"[GUI] 默认模型加载失败: {e}")

    def load_custom_model(self):
        if YOLO is None:
            messagebox.showerror("错误", "ultralytics 未安装")
            return
        p = filedialog.askopenfilename(
            filetypes=[("YOLO Weights", "*.pt")],
            title="选择模型权重 .pt"
        )
        if not p:
            return
        if not self._wait_for_image_worker():
            return
        try:
            restart_video = self.video_running
            if self.video_worker and self.video_worker.is_alive() and not self.stop_video():
                return
            self._register_model(p)
            if restart_video:
                self.start_video_inference()
            else:
                self.process_current_image_async()
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：\n{e}")


    def on_model_change(self, event):
        if self.model_var.get() in self.models:
            if not self._wait_for_image_worker():
                current_name = next(
                    (name for name, model in self.models.items() if model is self.current_model),
                    "",
                )
                self.model_var.set(current_name)
                return
            restart_video = self.video_running
            if self.video_worker and self.video_worker.is_alive() and not self.stop_video():
                current_name = next(
                    (name for name, model in self.models.items() if model is self.current_model),
                    "",
                )
                self.model_var.set(current_name)
                return
            self.current_model = self.models[self.model_var.get()]
            self._refresh_source_readiness()
            if restart_video:
                self.start_video_inference()
            else:
                self.process_current_image_async()

    def next_img(self):
        if self.data_manager and self.data_manager.current_data:
            self.curr_img_idx = (self.curr_img_idx + 1) % len(self.data_manager.current_data)
            self.process_current_image_async()

    def prev_img(self):
        if self.data_manager and self.data_manager.current_data:
            self.curr_img_idx = (self.curr_img_idx - 1) % len(self.data_manager.current_data)
            self.process_current_image_async()

    def process_current_image_async(self):
        if self._closing:
            return
        if self.video_running or (self.video_worker and self.video_worker.is_alive()):
            self._image_pending = False
            self.log("[GUI] 实时视频运行时不执行数据集图片推理")
            return
        if self._worker and self._worker.is_alive():
            self._image_pending = True
            return
        try:
            predict_settings = self._read_predict_settings()
        except ValueError as error:
            messagebox.showerror("参数错误", str(error))
            return
        self._image_pending = False
        self._worker = threading.Thread(
            target=self._image_worker_entry,
            args=(predict_settings,),
            daemon=True,
        )
        self._worker.start()

    def _image_worker_entry(self, predict_settings):
        try:
            self.process_current_image(predict_settings)
        except Exception as error:
            self.log(f"[GUI] 图片推理失败: {error}")
        finally:
            self._worker = None
            if self._image_pending:
                self._post_ui(self.process_current_image_async)

    def process_current_image(self, predict_settings=None):
        with self._lock:
            if not self.data_manager or not self.data_manager.current_data:
                return
            index = self.curr_img_idx
            item = self.data_manager.get_item(index)
            if not item:
                return
            rgb, ir, gt_boxes, fname = item
            self._infer_and_update(
                rgb,
                ir,
                gt_boxes,
                fname,
                (index + 1, len(self.data_manager.current_data)),
                predict_settings,
            )

    def _draw_gt(self, image, boxes):
        h, w = image.shape[:2]
        c = 0
        for b in boxes or []:
            if len(b) < 4:
                continue
            cx, cy, bw, bh = map(float, b[:4])
            if cx <= 1.0 and bw <= 1.0:
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            else:
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            c += 1
        return c

    def _predict_once(
        self,
        input_tensor,
        settings: dict,
        device: ComputeDevice,
        tracking: bool = False,
    ):
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        predict_kwargs = {
            "device": device.predict_arg,
            "half": device.use_half,
            "imgsz": settings["imgsz"],
            "conf": settings["conf"],
            "iou": settings["iou"],
            "classes": [0],
            "verbose": False,
        }
        if tracking:
            if not getattr(self.current_model, "_gui_tracker_registered", False):
                from ultralytics.trackers import register_tracker

                register_tracker(self.current_model, persist=True)
                self.current_model._gui_tracker_registered = True
            predict_kwargs.update(
                {
                    "mode": "track",
                    "tracker": str(BUNDLE_ROOT / "configs" / "bytetrack_realtime.yaml"),
                }
            )
        with torch.inference_mode():
            results = self.current_model.predict(input_tensor, **predict_kwargs)
        synchronize_if_cuda(device)
        return results, (time.perf_counter() - t0) * 1000.0

    def _draw_pred(
        self,
        image,
        input_tensor,
        predict_settings=None,
        *,
        tracking: bool = False,
        draw: bool = True,
    ):
        if self.current_model is None:
            return 0.0, 0, [], []
        settings = predict_settings or {"imgsz": 384, "conf": 0.25, "iou": 0.45}
        device = self.active_device
        try:
            results, dt = self._predict_once(input_tensor, settings, device, tracking=tracking)
        except Exception as e:
            if not device.is_cuda or not is_cuda_runtime_error(e):
                raise
            self.log(f"[ACCEL] GPU推理失败，自动回退CPU: {e}")
            clear_cuda_cache()
            self.current_model.predictor = None
            self.active_device = resolve_compute_device("cpu", self.cpu_threads)
            self._post_ui(self.device_var.set, "cpu")
            results, dt = self._predict_once(
                input_tensor,
                settings,
                self.active_device,
                tracking=tracking,
            )

        detections = []
        det_logs = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                score = float(b.conf[0])
                track_id = None
                if getattr(b, "id", None) is not None:
                    track_id = int(b.id[0].item())
                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "confidence": score,
                        "track_id": track_id,
                    }
                )
                id_text = f" id={track_id}" if track_id is not None else ""
                det_logs.append(
                    f"#{len(detections)}{id_text} conf={score:.2f} box=({x1},{y1},{x2},{y2})"
                )
        if draw:
            self._draw_detections(image, detections)
        return dt, len(detections), det_logs, detections

    @staticmethod
    def _draw_detections(image, detections: list[dict], inside_ids: set[int] | None = None):
        inside_ids = inside_ids or set()
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            score = float(detection["confidence"])
            track_id = detection.get("track_id")
            inside = bool(detection.get("inside_alert_roi", False)) or (
                track_id is not None and int(track_id) in inside_ids
            )
            color = (0, 0, 255) if not inside else (0, 165, 255)
            thickness = 3 if inside else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{track_id} {score:.2f}" if track_id is not None else f"{score:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )

    def _evaluate_alert(self, detections: list[dict], frame_shape) -> tuple[set[int], bool, int]:
        height, width = frame_shape[:2]
        track_ids = {
            int(detection["track_id"])
            for detection in detections
            if detection.get("track_id") is not None
        }
        self._seen_track_ids.update(track_ids)
        inside_ids = set()
        anonymous_inside_count = 0

        for detection in detections:
            detection["inside_alert_roi"] = False

        alert_roi = self.alert_roi
        if self.video_alert_enabled and alert_roi is not None:
            rx1, ry1, rx2, ry2 = alert_roi
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                # Ground-plane ROIs (doors, sidewalks, fence lines) should use the
                # pedestrian foot point; the box centre tends to trigger too early.
                foot_x = ((x1 + x2) / 2) / max(width, 1)
                foot_y = y2 / max(height, 1)
                if rx1 <= foot_x <= rx2 and ry1 <= foot_y <= ry2:
                    detection["inside_alert_roi"] = True
                    track_id = detection.get("track_id")
                    if track_id is None:
                        anonymous_inside_count += 1
                    else:
                        inside_ids.add(int(track_id))

        new_ids = inside_ids - self._inside_track_ids
        anonymous_entries = max(0, anonymous_inside_count - self._anonymous_inside_count)
        trigger = bool(new_ids) or anonymous_entries > 0
        if trigger:
            self._alert_entry_count += len(new_ids) + anonymous_entries
            self._alert_flash_until = time.monotonic() + 1.2
            entered_parts = [",".join(map(str, sorted(new_ids)))] if new_ids else []
            if anonymous_entries:
                entered_parts.append(f"未跟踪目标x{anonymous_entries}")
            entered = " / ".join(entered_parts)
            self.log(f"[ALERT] 行人进入警戒区域: {entered}")
            self._play_alarm_sound()

        self._inside_track_ids = inside_ids
        self._anonymous_inside_count = anonymous_inside_count
        current_count = len(detections)
        if self.video_tracking_enabled:
            anonymous_current = sum(detection.get("track_id") is None for detection in detections)
            self._anonymous_seen_count = max(
                self._anonymous_seen_count,
                len(self._seen_track_ids) + anonymous_current,
            )
        else:
            self._anonymous_seen_count += max(0, current_count - self._anonymous_scene_count)
        self._anonymous_scene_count = current_count
        unique_count = self._anonymous_seen_count
        self._post_ui(
            self._set_count_metrics,
            len(detections),
            unique_count,
            self._alert_entry_count,
        )
        return inside_ids, trigger, unique_count

    def _draw_alert_roi(self, image):
        alert_roi = self.alert_roi
        if alert_roi is None:
            return
        height, width = image.shape[:2]
        x1 = int(alert_roi[0] * width)
        y1 = int(alert_roi[1] * height)
        x2 = int(alert_roi[2] * width)
        y2 = int(alert_roi[3] * height)
        active = time.monotonic() < self._alert_flash_until
        color = (0, 0, 255) if active else (0, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if active else 2)
        cv2.putText(
            image,
            "ALERT ROI",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    def _play_alarm_sound(self):
        def play():
            try:
                if os.name == "nt":
                    import winsound

                    winsound.Beep(1250, 180)
                    winsound.Beep(1550, 180)
                else:
                    self._post_ui(self.root.bell)
            except Exception:
                pass

        threading.Thread(target=play, daemon=True).start()

    def _infer_and_update(self, rgb, ir, gt_boxes, fname, total, predict_settings=None):
        ir_show = to_uint8(ir if ir.ndim == 3 else cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR))
        t0 = time.perf_counter()
        ir_preprocessed = self.fusion_engine.preprocess_ir(ir)
        fused = self.fusion_engine.for_display_with_preprocessed_ir(rgb, ir_preprocessed)
        model_input = self._model_input_with_preprocessed_ir(rgb, ir_preprocessed)
        fuse_ms = (time.perf_counter() - t0) * 1000.0
        res = fused.copy()
        gt_count = self._draw_gt(res, gt_boxes)
        det_ms, pred_count, det_logs, _detections = (0.0, 0, [], [])
        try:
            if model_input is not None:
                det_ms, pred_count, det_logs, _detections = self._draw_pred(
                    res,
                    model_input,
                    predict_settings,
                )
        except Exception as e:
            self.log(f"[GUI] 推理出错: {e}")
        if det_logs:
            self.log("[PRED] " + "; ".join(det_logs))
        self.latest_result_base_frame = res.copy()
        self._draw_alert_roi(res)
        self.latest_result_frame = res.copy()
        self._post_ui(
            self._update_ui_images,
            rgb,
            ir_show,
            fused,
            res,
            fname,
            total,
            fuse_ms,
            det_ms,
            gt_count,
            pred_count,
        )

    def _update_ui_images(
        self,
        rgb,
        ir,
        fused,
        res,
        fname,
        total,
        fuse_ms,
        det_ms,
        gt_count,
        pred_count,
        runtime_fps=0.0,
    ):
        for img, label in [
            (rgb, self.lbl_rgb.inner_label),
            (ir, self.lbl_ir.inner_label),
            (fused, self.lbl_fusion.inner_label),
            (res, self.lbl_result.inner_label),
        ]:
            self._show_image(img, label)
        m = self.model_var.get() if self.current_model else "未加载"
        mode = f"{self._current_model_channels()}ch" if self.current_model else "-"
        if isinstance(total, tuple):
            idx_text = f"[{total[0]}/{total[1]}]"
        elif isinstance(total, str):
            idx_text = f"[{total}]"
        else:
            idx_text = f"[{self.curr_img_idx + 1}/{total}]"
        fps_text = f" FPS:{runtime_fps:.1f}" if runtime_fps > 0 else ""
        self.info_var.set(
            f"{idx_text} {fname} | 模型:{m} ({mode}) | "
            f"设备:{self.active_device.label} | Fuse:{fuse_ms:.1f}ms "
            f"Det:{det_ms:.1f}ms{fps_text} | GT:{gt_count} Pred:{pred_count}"
        )
        if hasattr(self, "fps_metric_var"):
            self.fps_metric_var.set(f"{runtime_fps:.1f}" if runtime_fps > 0 else "--")
        if hasattr(self, "latency_metric_var"):
            self.latency_metric_var.set(f"{det_ms:.1f} ms" if det_ms > 0 else "--")
        if time.monotonic() < self._alert_flash_until:
            self._set_run_state("警戒触发", "warning")
        elif self.video_running:
            self._set_run_state("已暂停" if self.video_paused else "检测中", "paused" if self.video_paused else "running")

    def _schedule_video_ui_update(self, *args):
        with self._video_ui_lock:
            self._video_ui_payload = args

    def _show_image(self, cv_img, tk_label):
        if cv_img is None:
            tk_label.config(text="暂无画面", image="")
            self._last_view_images.pop(tk_label, None)
            return
        self._last_view_images[tk_label] = cv_img
        cv_img = to_uint8(cv_img)
        ww, wh = max(160, tk_label.winfo_width() - 4), max(120, tk_label.winfo_height() - 4)
        h, w = cv_img.shape[:2]
        s = min(ww / max(w, 1), wh / max(h, 1))
        render_w, render_h = max(1, int(w * s)), max(1, int(h * s))
        img = cv2.resize(cv_img, (render_w, render_h))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        tk_label.config(image=photo, text="")
        self._img_refs[tk_label] = photo
        label_w, label_h = max(1, tk_label.winfo_width()), max(1, tk_label.winfo_height())
        self._display_geometry[tk_label] = (
            w,
            h,
            render_w,
            render_h,
            max(0, (label_w - render_w) / 2),
            max(0, (label_h - render_h) / 2),
        )

    def _set_no_image_all(self):
        self.latest_result_frame = None
        self.latest_result_base_frame = None
        for frame in [self.lbl_rgb, self.lbl_ir, self.lbl_fusion, self.lbl_result]:
            frame.inner_label.config(text="暂无画面", image="")
            self._img_refs.pop(frame.inner_label, None)
            self._last_view_images.pop(frame.inner_label, None)
            self._display_geometry.pop(frame.inner_label, None)

    def open_visible_video(self):
        p = filedialog.askopenfilename(
            title="选择可见光视频",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")],
        )
        if p:
            self._camera_scan_generation += 1
            was_camera = self.stream_mode.startswith("camera_")
            if not self.stop_video():
                return
            self.stream_mode = "video"
            if was_camera:
                self.ir_video_source = None
            self.video_source = p
            self.video_backend = None
            self.log(f"[GUI] 可见光视频: {p}")
            self._set_run_state("输入已选择", "idle")
            self._refresh_source_readiness()

    def open_ir_video(self):
        p = filedialog.askopenfilename(
            title="选择红外视频",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")],
        )
        if p:
            self._camera_scan_generation += 1
            was_camera = self.stream_mode.startswith("camera_")
            if not self.stop_video():
                return
            self.stream_mode = "video"
            if was_camera:
                self.video_source = None
            self.ir_video_source = p
            self.ir_video_backend = None
            self.log(f"[GUI] 红外视频: {p}")
            self._set_run_state("输入已选择", "idle")
            self._refresh_source_readiness()

    def open_auto_camera(self):
        if self.camera_scan_worker and self.camera_scan_worker.is_alive():
            self.log("[CAM] 正在扫描摄像头，请稍候")
            return
        if not self._wait_for_image_worker() or not self.stop_video():
            return
        self._camera_scan_generation += 1
        generation = self._camera_scan_generation
        self.info_var.set("正在扫描摄像头...")
        self._set_run_state("扫描摄像头", "warning")
        self.log("[CAM] 开始自动识别摄像头")
        self.camera_scan_worker = threading.Thread(
            target=self._scan_cameras,
            args=(generation,),
            daemon=True,
        )
        self.camera_scan_worker.start()

    def _scan_cameras(self, generation: int):
        try:
            devices = discover_cameras(validate=True)
            self._post_ui(self._start_auto_camera, generation, devices)
        except Exception as e:
            self._post_ui(self._show_camera_scan_error, generation, str(e))

    def _show_camera_scan_error(self, generation: int, error: str):
        if generation != self._camera_scan_generation:
            return
        self.info_var.set("摄像头扫描失败")
        self._set_run_state("扫描失败", "error")
        self.log(f"[CAM] 摄像头扫描失败: {error}")
        messagebox.showerror("摄像头错误", f"自动识别摄像头失败：\n{error}")

    def _start_auto_camera(self, generation: int, devices: list[CameraDevice]):
        if generation != self._camera_scan_generation or self._closing:
            return
        if not self._wait_for_image_worker():
            return
        if not devices:
            self.info_var.set("未检测到可用摄像头")
            self._set_run_state("未发现摄像头", "warning")
            messagebox.showwarning(
                "未检测到摄像头",
                "没有检测到可用摄像头。请检查Windows相机权限、连接状态，或关闭正在占用摄像头的程序。",
            )
            return

        for device in devices:
            self.log(f"[CAM] 设备{device.index}: {device.name} | 类型:{device.kind}")

        setup = choose_auto_camera_setup(devices)
        if setup is None:
            self.info_var.set("没有可用的RGB摄像头")
            messagebox.showwarning("摄像头不适用", "检测到了摄像头，但无法确定可见光输入设备。")
            return

        channels = 6 if setup.mode == "rgbt" else 3
        if not self._ensure_model_channels(channels):
            messagebox.showerror("缺少模型", f"没有找到可用的{channels}通道模型。")
            return

        self.video_source = setup.rgb.index
        self.video_backend = setup.rgb.backend
        if setup.ir is not None:
            self.stream_mode = "camera_rgbt"
            self.ir_video_source = setup.ir.index
            self.ir_video_backend = setup.ir.backend
            self.log(
                f"[CAM] 自动使用RGB+IR: RGB={setup.rgb.name}({setup.rgb.index}), "
                f"IR={setup.ir.name}({setup.ir.index}) | 6通道模型:{self.model_var.get()}"
            )
        else:
            self.stream_mode = "camera_rgb"
            self.ir_video_source = None
            self.ir_video_backend = None
            if setup.rgb.is_integrated:
                source_type = "内置前置"
            elif setup.rgb.is_generic:
                source_type = "默认RGB"
            else:
                source_type = "外接"
            self.log(
                f"[CAM] 自动使用{source_type}摄像头: {setup.rgb.name}({setup.rgb.index}) | "
                f"3通道模型:{self.model_var.get()}"
            )
        self._set_run_state("摄像头已就绪", "idle")
        self._refresh_source_readiness()

    def open_rgbt_cameras(self):
        if not self._wait_for_image_worker() or not self.stop_video():
            return
        self._camera_scan_generation += 1
        rgb_camera_id = simpledialog.askinteger(
            "RGB+IR双摄像头",
            "请输入可见光摄像头编号：",
            initialvalue=0,
            minvalue=0,
            parent=self.root,
        )
        if rgb_camera_id is None:
            return
        ir_camera_id = simpledialog.askinteger(
            "RGB+IR双摄像头",
            "请输入红外摄像头编号（通常第二个摄像头为1）：",
            initialvalue=1,
            minvalue=0,
            parent=self.root,
        )
        if ir_camera_id is None:
            return
        if rgb_camera_id == ir_camera_id:
            messagebox.showerror("摄像头编号错误", "可见光和红外摄像头必须使用不同的编号。")
            return
        if not self._ensure_model_channels(6):
            messagebox.showerror(
                "缺少模型",
                "没有找到6通道模型。请手动加载训练好的 RGBT best.pt。"
            )
            return
        self.stream_mode = "camera_rgbt"
        self.video_source = rgb_camera_id
        self.ir_video_source = ir_camera_id
        self.video_backend = cv2.CAP_DSHOW if os.name == "nt" else None
        self.ir_video_backend = cv2.CAP_DSHOW if os.name == "nt" else None
        self.log(
            f"[CAM] 可见光摄像头: {rgb_camera_id} | 红外摄像头: {ir_camera_id} | "
            f"6通道模型: {self.model_var.get()}"
        )
        self._set_run_state("双摄已就绪", "idle")
        self._refresh_source_readiness()

    def toggle_pause_video(self):
        if self.video_running:
            self.video_paused = not self.video_paused
            if self.video_paused:
                self.event_recorder.flush()
            else:
                self._fps_times.clear()
                self._fps_ema = 0.0
            self.log("[GUI] 视频已暂停" if self.video_paused else "[GUI] 视频继续播放")
            self._set_run_state("已暂停" if self.video_paused else "检测中", "paused" if self.video_paused else "running")

    def stop_video(self, wait_timeout: float | None = 10.0) -> bool:
        self.video_running = False
        self.video_paused = False
        worker = self.video_worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=wait_timeout)
        if worker is not None and worker.is_alive():
            self.log("[GUI] 视频线程仍在结束，已取消本次重启操作")
            return False
        self.video_worker = None
        self._release_video_io()
        if hasattr(self, "run_state_var"):
            self._on_video_stopped()
        return True

    def _release_video_io(self):
        for cap in (self.cap_vis, self.cap_ir):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        self.cap_vis = None
        self.cap_ir = None
        self._close_output_writer(clear_request=True)

    def _close_output_writer(self, clear_request: bool = False):
        if self.output_writer is not None:
            try:
                self.output_writer.release()
            except Exception:
                pass
        self.output_writer = None
        self.output_writer_path = None
        if clear_request:
            self.output_path = None

    def _write_output_frame(self, frame, source_fps: float, is_camera: bool) -> None:
        requested_path = self.output_path
        if not requested_path:
            return
        if self.output_writer is None or self.output_writer_path != requested_path:
            self._close_output_writer(clear_request=False)
            output = Path(requested_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            writer_fps = source_fps
            if is_camera and self._fps_ema > 1.0:
                writer_fps = min(source_fps, self._fps_ema)
            writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, float(writer_fps)),
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                if self.output_path == requested_path:
                    self.output_path = None
                self.log(f"[GUI] 无法创建视频结果文件: {output}")
                return
            self.output_writer = writer
            self.output_writer_path = requested_path
            self.log(f"[GUI] 已开始录制视频结果: {output}")
        self.output_writer.write(frame)

    @staticmethod
    def _open_capture(source, backend=None):
        if isinstance(source, int) and os.name == "nt":
            cap = cv2.VideoCapture(source, backend or cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start_video_inference(self):
        if self.current_model is None:
            messagebox.showwarning("提示", "请先加载模型")
            return
        if not self._wait_for_image_worker():
            return

        ch = self._current_model_channels()
        self._set_stream_titles(ch >= 6)

        if self.stream_mode == "camera_rgb" and ch != 3:
            messagebox.showwarning("提示", "可见光摄像头模式必须使用3通道模型")
            return

        if self.stream_mode == "camera_rgbt" and ch < 6:
            messagebox.showwarning("提示", "RGB+IR摄像头模式必须使用6通道模型")
            return

        if self.video_source is None:
            messagebox.showwarning("提示", "请先选择可见光视频")
            return

        if ch >= 6 and self.ir_video_source is None:
            messagebox.showwarning("提示", "6通道模型请同时选择红外视频")
            return

        if ch >= 6 and self.video_source == self.ir_video_source:
            messagebox.showwarning(
                "提示",
                "当前可见光视频和红外视频是同一个文件。\n"
                "6通道模型必须使用真实配对的 RGB/IR 视频，当前不允许开始推理。"
            )
            return

        try:
            predict_settings = self._read_predict_settings()
            video_speed = self._read_video_speed()
        except ValueError as error:
            messagebox.showerror("参数错误", str(error))
            return
        if not self.stop_video():
            return
        self.video_model_channels = ch
        self.video_model_name = self.model_var.get()
        self.video_speed = video_speed
        self.video_performance_mode = self._performance_key()
        self.video_predict_settings = self._video_profile_settings(
            predict_settings,
            self.video_performance_mode,
        )
        self.video_tracking_enabled = bool(self.tracking_var.get())
        self.video_alert_enabled = bool(self.alert_var.get())
        self.video_ir_frame_offset = int(self.settings.get("ir_frame_offset", 0))
        self.video_ir_shift = (
            int(self.settings.get("ir_shift_x", 0)),
            int(self.settings.get("ir_shift_y", 0)),
        )
        self.event_recorder.configure(
            self.settings.get("pre_event_seconds", 3.0),
            self.settings.get("post_event_seconds", 5.0),
            self.settings.get("alarm_cooldown_seconds", 5.0),
        )
        self.event_recorder.reset()
        self._reset_tracking_state()
        self._save_settings()
        self.video_running = True
        self._set_run_state("检测中", "running")
        self._refresh_source_readiness()
        with self._video_ui_lock:
            self._video_ui_payload = None
        self._fps_ema = 0.0
        self._fps_times.clear()
        self.log(
            f"[ACCEL] 视频推理使用 {self.active_device.label} | "
            f"imgsz={self.video_predict_settings['imgsz']} | "
            f"模式={self.performance_var.get()} | 跟踪={'开' if self.video_tracking_enabled else '关'}"
        )
        self.video_worker = threading.Thread(target=self._video_loop, daemon=True)
        self.video_worker.start()


    def _video_loop(self):
        try:
            ch = self.video_model_channels
            use_ir = ch >= 6
            is_camera = self.stream_mode.startswith("camera_")

            if self.video_source is None:
                raise RuntimeError("请先选择可见光视频")

            if use_ir and self.ir_video_source is None:
                raise RuntimeError("请先选择红外视频")

            self.cap_vis = self._open_capture(self.video_source, self.video_backend)
            self.cap_ir = self._open_capture(self.ir_video_source, self.ir_video_backend) if use_ir else None
            if self.cap_vis is None or (use_ir and self.cap_ir is None):
                raise RuntimeError("视频对象创建失败，请检查输入源")
            if not self.cap_vis.isOpened() or (use_ir and not self.cap_ir.isOpened()):
                raise RuntimeError("视频或摄像头打开失败")

            fps = self.cap_vis.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 1.0 or fps > 240.0:
                fps = 25.0
            if use_ir and not is_camera and self.video_ir_frame_offset != 0:
                target = self.cap_ir if self.video_ir_frame_offset > 0 else self.cap_vis
                for _ in range(abs(self.video_ir_frame_offset)):
                    if not target.grab():
                        break
                self.log(f"[SYNC] 已应用IR帧偏移: {self.video_ir_frame_offset}")
            if is_camera:
                total_frames = "实时"
            elif use_ir:
                visible_frames = int(self.cap_vis.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                infrared_frames = int(self.cap_ir.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if self.video_ir_frame_offset > 0:
                    infrared_frames = max(0, infrared_frames - self.video_ir_frame_offset)
                elif self.video_ir_frame_offset < 0:
                    visible_frames = max(0, visible_frames + self.video_ir_frame_offset)
                total_frames = min(visible_frames, infrared_frames) or 1
            else:
                total_frames = int(self.cap_vis.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
            delay = max(0.001, 1.0 / (fps * self.video_speed))
            skip_accum = 0.0
            inference_stride = 2 if self.video_performance_mode == "smooth" and use_ir else 1

            while self.video_running:
                frame_t0 = time.perf_counter()
                if self.video_paused:
                    time.sleep(0.05)
                    continue

                if is_camera and use_ir:
                    if not self.cap_vis.grab() or not self.cap_ir.grab():
                        break
                    rv, fv = self.cap_vis.retrieve()
                    ri, fi = self.cap_ir.retrieve()
                    if not rv or not ri:
                        break
                else:
                    rv, fv = self.cap_vis.read()
                    if not rv:
                        break
                    if use_ir:
                        ri, fi = self.cap_ir.read()
                        if not ri:
                            break
                    else:
                        fi = None

                visible_frame_no = int(self.cap_vis.get(cv2.CAP_PROP_POS_FRAMES))
                frame_no = max(1, visible_frame_no + min(0, self.video_ir_frame_offset))
                self._video_frame_index += 1
                should_infer = (
                    self._video_frame_index % inference_stride == 1 % inference_stride
                    or not self._has_inference_result
                )
                fuse_t0 = time.perf_counter()
                if use_ir:
                    fi = self.fusion_engine.shift_ir(fi, *self.video_ir_shift)
                    ir_show = to_uint8(fi if fi.ndim == 3 else cv2.cvtColor(fi, cv2.COLOR_GRAY2BGR))
                    if should_infer:
                        preprocess_max_side = self.IR_PREPROCESS_MAX_SIDE[
                            self.video_performance_mode
                        ]
                        ir_preprocessed = self.fusion_engine.preprocess_ir(
                            fi,
                            realtime=preprocess_max_side is not None,
                            realtime_max_side=preprocess_max_side or max(fi.shape[:2]),
                        )
                        fused = self.fusion_engine.for_display_with_preprocessed_ir(fv, ir_preprocessed)
                        model_input = self.fusion_engine.for_model_with_preprocessed_ir(fv, ir_preprocessed)
                    else:
                        fused = self.fusion_engine.for_display_fast(fv, fi)
                        model_input = None
                else:
                    ir_show = None
                    fused = fv.copy()
                    model_input = fv
                fuse_ms = (time.perf_counter() - fuse_t0) * 1000.0
                res = fused.copy()
                if should_infer:
                    det_ms, pred_count, _det_logs, detections = self._draw_pred(
                        res,
                        model_input,
                        self.video_predict_settings,
                        tracking=self.video_tracking_enabled,
                        draw=False,
                    )
                    self._last_detections = detections
                    self._has_inference_result = True
                else:
                    det_ms = 0.0
                    detections = list(self._last_detections)
                    pred_count = len(detections)

                inside_ids, alert_triggered, unique_count = self._evaluate_alert(
                    detections,
                    res.shape,
                )
                self._draw_detections(res, detections, inside_ids)
                self.latest_result_base_frame = res.copy()
                self._draw_alert_roi(res)
                self.latest_result_frame = res.copy()

                self._write_output_frame(res, fps, is_camera)

                elapsed = time.perf_counter() - frame_t0
                self._fps_times.append(time.perf_counter())
                if len(self._fps_times) > 1:
                    self._fps_ema = (len(self._fps_times) - 1) / max(
                        self._fps_times[-1] - self._fps_times[0],
                        1e-6,
                    )
                else:
                    self._fps_ema = 1.0 / max(elapsed, 1e-6)
                source_name = "camera" if is_camera else Path(str(self.video_source)).name
                if should_infer:
                    self.detection_recorder.append_frame(
                        source=source_name,
                        frame="实时" if is_camera else frame_no,
                        detections=detections,
                        inside_ids=inside_ids,
                        current_count=pred_count,
                        unique_count=unique_count,
                        device=self.active_device.label,
                        fps=self._fps_ema,
                    )
                if (
                    self.video_alert_enabled and self.alert_roi is not None
                ) or self.event_recorder.is_active:
                    started = self.event_recorder.add_frame(
                        res,
                        fps=self._fps_ema,
                        trigger=bool(
                            alert_triggered
                            and self.video_alert_enabled
                            and self.alert_roi is not None
                        ),
                        metadata={
                            "source": source_name,
                            "frame": "实时" if is_camera else frame_no,
                            "track_ids": sorted(inside_ids),
                            "device": self.active_device.label,
                            "model": self.video_model_name,
                        },
                    )
                    if started:
                        self.log("[ALERT] 已开始保存报警前后视频")
                self._schedule_video_ui_update(
                    fv,
                    ir_show,
                    fused,
                    res,
                    "camera" if is_camera else "video",
                    "实时" if is_camera else (frame_no, total_frames),
                    fuse_ms,
                    det_ms,
                    0,
                    pred_count,
                    self._fps_ema,
                )
                loop_elapsed = time.perf_counter() - frame_t0
                if loop_elapsed < delay:
                    time.sleep(delay - loop_elapsed)
                elif (
                    not is_camera
                    and self.video_performance_mode != "smooth"
                    and not self.output_path
                ):
                    skip_accum += loop_elapsed / delay - 1.0
                    skip_frames = min(8, int(skip_accum))
                    if skip_frames > 0:
                        skip_accum -= skip_frames
                        for _ in range(skip_frames):
                            rv_skip = self.cap_vis.grab()
                            ri_skip = self.cap_ir.grab() if use_ir else True
                            if not rv_skip or not ri_skip:
                                break
        except Exception as e:
            error_message = str(e)
            if self.video_running:
                self.log(f"[GUI] 视频推理失败: {e}")
        finally:
            self.video_running = False
            self.event_recorder.flush()
            self._release_video_io()
            if self.video_worker is threading.current_thread():
                self.video_worker = None
            self._post_ui(self._on_video_stopped, locals().get("error_message"))

    def save_current_result(self):
        if self.latest_result_frame is None:
            messagebox.showwarning("提示", "当前没有可保存的结果")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        p = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".jpg",
            initialdir=str(self.save_dir),
            initialfile=f"result_{ts}.jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("MP4", "*.mp4")],
        )
        if not p:
            return
        if Path(p).suffix.lower() == ".mp4":
            if not self.video_running:
                messagebox.showwarning("提示", "请先启动视频或摄像头推理，再选择MP4录制路径")
                return
            output_key = os.path.normcase(os.path.abspath(p))
            input_keys = {
                os.path.normcase(os.path.abspath(str(source)))
                for source in (self.video_source, self.ir_video_source)
                if isinstance(source, (str, os.PathLike))
            }
            if output_key in input_keys:
                messagebox.showerror("保存失败", "输出视频不能覆盖正在读取的RGB或IR输入视频")
                return
            self.output_path = p
            self.log(f"[GUI] 视频结果将保存到: {p}")
        else:
            image = to_uint8(self.latest_result_frame)
            if image is None or not cv2.imwrite(p, image):
                messagebox.showerror("保存失败", f"无法写入图片：\n{p}")
                return
            self.log(f"[GUI] 图片结果已保存到: {p}")

    def _on_close(self):
        self._save_settings()
        self._closing = True
        self._camera_scan_generation += 1
        self.stop_video(wait_timeout=None)
        self._wait_for_image_worker(timeout=None)
        self.event_recorder.close()
        self.root.destroy()


def main_ui_start():
    root = tk.Tk()
    try:
        root.state("zoomed")
    except Exception:
        pass
    MainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main_ui_start()
