from __future__ import annotations

import os
import sys
import time
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
OUTPUT_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NightPedRGBT"
    if getattr(sys, "frozen", False)
    else Path(PROJECT_ROOT)
)

from app_utils.camera import CameraDevice, choose_auto_camera_setup, discover_cameras
from app_utils.data_loader import DatasetManager
from app_utils.fusion import FusionEngine

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class MainApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("夜间行人检测系统 (RGB 3ch / RGBT 6ch + YOLO11)")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 700)

        self.data_manager = None
        self.fusion_engine = FusionEngine()
        self.models = {}
        self.model_channels = {}
        self.current_model = None
        self.curr_img_idx = 0

        self.path_var = tk.StringVar(value="")
        self.split_var = tk.StringVar(value="test")
        self.model_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="cpu")
        self.imgsz_var = tk.IntVar(value=384)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.info_var = tk.StringVar(value="请先选择 LLVIP 路径并加载模型")

        self._lock = threading.Lock()
        self._worker = None
        self._img_refs = {}
        self._video_ui_pending = False

        self.video_worker = None
        self.video_running = False
        self.video_paused = False
        self.video_source = None
        self.ir_video_source = None
        self.video_backend = None
        self.ir_video_backend = None
        self.stream_mode = "video"
        self.camera_scan_worker = None
        self.cap_vis = None
        self.cap_ir = None
        self.output_writer = None
        self.output_path = None
        self.latest_result_frame = None
        self.save_dir = OUTPUT_ROOT / "runs" / "gui_results"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._load_default_model_if_exist()
        self._set_no_image_all()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def log(self, s: str):
        print(s)
        try:
            self.root.after(0, lambda: self._append_log(str(s)))
        except Exception:
            pass

    def _append_log(self, s: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", s + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#eaeaea", pady=6)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, text="LLVIP路径:", bg="#eaeaea").pack(side=tk.LEFT, padx=6)
        self.path_entry = tk.Entry(top, textvariable=self.path_var, width=42)
        self.path_entry.pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="浏览", command=self.browse_dataset).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="加载/刷新", command=self.load_dataset_from_entry).pack(side=tk.LEFT, padx=6)

        tk.Label(top, text="子集:", bg="#eaeaea").pack(side=tk.LEFT, padx=6)
        self.split_combo = ttk.Combobox(
            top,
            textvariable=self.split_var,
            values=["train", "test", "val"],
            width=8,
            state="readonly",
        )
        self.split_combo.pack(side=tk.LEFT, padx=4)
        self.split_combo.bind("<<ComboboxSelected>>", self.on_split_change)

        tk.Label(top, text="模型:", bg="#eaeaea").pack(side=tk.LEFT, padx=6)
        self.model_combo = ttk.Combobox(
            top,
            textvariable=self.model_var,
            values=[],
            width=22,
            state="readonly",
        )
        self.model_combo.pack(side=tk.LEFT, padx=4)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        tk.Button(top, text="+ 加载模型(.pt)", command=self.load_custom_model).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="device:", bg="#eaeaea").pack(side=tk.LEFT, padx=6)
        ttk.Combobox(
            top,
            textvariable=self.device_var,
            values=["cpu", "0", "1"],
            width=6,
            state="readonly",
        ).pack(side=tk.LEFT, padx=4)

        for text, var in (("imgsz", self.imgsz_var), ("conf", self.conf_var), ("iou", self.iou_var)):
            tk.Label(top, text=f"{text}:", bg="#eaeaea").pack(side=tk.LEFT, padx=6)
            tk.Entry(top, textvariable=var, width=6).pack(side=tk.LEFT, padx=4)

        mid = tk.Frame(self.root, bg="#fff")
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        for i in range(2):
            mid.grid_rowconfigure(i, weight=1, uniform="r")
            mid.grid_columnconfigure(i, weight=1, uniform="c")

        self.lbl_rgb = self._box(mid, "Visible (RGB)")
        self.lbl_fusion = self._box(mid, "Fusion View")
        self.lbl_ir = self._box(mid, "Infrared (IR)")
        self.lbl_result = self._box(mid, "Result (绿=GT, 红=Pred)")

        self.lbl_rgb.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.lbl_fusion.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.lbl_ir.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.lbl_result.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

        btm = tk.Frame(self.root, bg="#f0f0f0", pady=6)
        btm.pack(side=tk.BOTTOM, fill=tk.X)
        items = [
            ("< 上一张", self.prev_img, 10),
            ("下一张 >", self.next_img, 10),
            ("可见光视频", self.open_visible_video, 12),
            ("红外视频", self.open_ir_video, 10),
            ("自动摄像头", self.open_auto_camera, 11),
            ("手动RGB+IR", self.open_rgbt_cameras, 12),
            ("暂停/继续", self.toggle_pause_video, 10),
            ("停止视频", self.stop_video, 10),
            ("保存结果", self.save_current_result, 10),
        ]
        for text, cmd, width in items:
            tk.Button(btm, text=text, command=cmd, width=width).pack(side=tk.LEFT, padx=5)

        tk.Label(btm, text="倍速", bg="#f0f0f0").pack(side=tk.LEFT, padx=(10, 4))
        tk.Spinbox(
            btm,
            from_=0.25,
            to=4.0,
            increment=0.25,
            textvariable=self.speed_var,
            width=6,
        ).pack(side=tk.LEFT, padx=4)

        tk.Label(
            btm,
            textvariable=self.info_var,
            bg="#f0f0f0",
            font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=12)

        logf = tk.Frame(self.root, bg="#f8f8f8")
        logf.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(logf, text="Log:", bg="#f8f8f8").pack(side=tk.LEFT, padx=6)
        self.log_text = tk.Text(logf, height=5)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=6)
        self.log_text.insert("end", "GUI Ready.\n")
        self.log_text.config(state="disabled")

    def _box(self, parent, title):
        f = tk.Frame(parent, bd=2, relief=tk.GROOVE, bg="#222")
        title_label = tk.Label(f, text=title, bg="#efefef", font=("Arial", 10, "bold"))
        title_label.pack(side=tk.TOP, fill=tk.X)
        l = tk.Label(f, text="No Image", bg="#333", fg="white")
        l.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        f.title_label = title_label
        f.inner_label = l
        return f

    def _set_stream_titles(self, use_ir: bool):
        self.lbl_rgb.title_label.config(text="Visible (RGB)")
        self.lbl_ir.title_label.config(text="Infrared (IR)" if use_ir else "Infrared (未使用)")
        self.lbl_fusion.title_label.config(text="Fusion View" if use_ir else "RGB Preview")
        self.lbl_result.title_label.config(text="Result (红=Pred)")

    def browse_dataset(self):
        p = filedialog.askdirectory(title="请选择 LLVIP 根目录(包含 visible/infrared)")
        if p:
            self.path_var.set(p)

    def load_dataset_from_entry(self):
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

    def _current_model_channels(self) -> int:
        name = self.model_var.get()
        return int(self.model_channels.get(name, 3))

    def _select_loaded_model_by_channels(self, channels: int) -> bool:
        for name, model in self.models.items():
            if int(self.model_channels.get(name, 3)) == channels:
                self.model_var.set(name)
                self.current_model = model
                self.log(f"[GUI] 已自动切换到{channels}通道模型: {name}")
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
                candidates.extend(runs_dir.rglob("best.pt"))
            candidates.sort(
                key=lambda p: (
                    int("rgbt" in str(p).lower()),
                    int("cbam" in str(p).lower()),
                    int("ft8002" in str(p).lower()),
                    p.stat().st_mtime,
                ),
                reverse=True,
            )

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
                    int("ft8002" in s),
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
        try:
            self._register_model(p)
            self.process_current_image_async()
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：\n{e}")


    def on_model_change(self, event):
        if self.model_var.get() in self.models:
            self.current_model = self.models[self.model_var.get()]
            ch = self._current_model_channels()
            if self.video_running and self.stream_mode == "camera_rgb" and ch != 3:
                self.stop_video()
                messagebox.showwarning(
                    "提示",
                    "单个可见光摄像头只能使用3通道模型，实时推理已停止。"
                )
                return
            if self.video_running and self.stream_mode == "camera_rgbt" and ch < 6:
                self.stop_video()
                messagebox.showwarning(
                    "提示",
                    "RGB+IR双摄像头需要6通道模型，实时推理已停止。"
                )
                return
            if self.video_running and ch >= 6:
                if self.video_source is None or self.ir_video_source is None:
                    self.stop_video()
                    messagebox.showwarning(
                        "提示",
                        "当前已切换到6通道模型，缺少红外视频，已自动停止视频推理。\n请先加载IR视频后再开始。"
                    )
                elif self.video_source == self.ir_video_source:
                    self.stop_video()
                    messagebox.showwarning(
                        "提示",
                        "当前已切换到6通道模型，但RGB和IR是同一个视频，已自动停止视频推理。\n请加载真实配对的IR视频后再开始。"
                    )
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
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self.process_current_image, daemon=True)
        self._worker.start()

    def process_current_image(self):
        with self._lock:
            if not self.data_manager or not self.data_manager.current_data:
                return
            item = self.data_manager.get_item(self.curr_img_idx)
            if not item:
                return
            rgb, ir, gt_boxes, fname = item
            self._infer_and_update(rgb, ir, gt_boxes, fname, len(self.data_manager.current_data))

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

    def _draw_pred(self, image, input_tensor):
        if self.current_model is None:
            return 0.0, 0, []
        t0 = time.time()
        results = self.current_model.predict(
            input_tensor,
            device=self.device_var.get(),
            imgsz=max(32, int(self.imgsz_var.get())),
            conf=max(0.01, float(self.conf_var.get())),
            iou=max(0.01, float(self.iou_var.get())),
            classes=[0],
            verbose=False,
        )
        dt = (time.time() - t0) * 1000.0
        c = 0
        det_logs = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                score = float(b.conf[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{score:.2f}"
                cv2.putText(
                    image,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                det_logs.append(f"#{c + 1} conf={score:.2f} box=({x1},{y1},{x2},{y2})")
                c += 1
        return dt, c, det_logs

    def _infer_and_update(self, rgb, ir, gt_boxes, fname, total):
        ir_show = ir if ir.ndim == 3 else cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        t0 = time.time()
        ir_preprocessed = self.fusion_engine.preprocess_ir(ir)
        fused = self.fusion_engine.for_display_with_preprocessed_ir(rgb, ir_preprocessed)
        model_input = self._model_input_with_preprocessed_ir(rgb, ir_preprocessed)
        fuse_ms = (time.time() - t0) * 1000.0
        res = fused.copy()
        gt_count = self._draw_gt(res, gt_boxes)
        det_ms, pred_count, det_logs = (0.0, 0, [])
        try:
            if model_input is not None:
                det_ms, pred_count, det_logs = self._draw_pred(res, model_input)
        except Exception as e:
            self.log(f"[GUI] 推理出错: {e}")
        if det_logs:
            self.log("[PRED] " + "; ".join(det_logs))
        self.latest_result_frame = res.copy()
        self.root.after(
            0,
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

    def _update_ui_images(self, rgb, ir, fused, res, fname, total, fuse_ms, det_ms, gt_count, pred_count):
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
        self.info_var.set(
            f"{idx_text} {fname} | 模型:{m} ({mode}) | "
            f"Fuse:{fuse_ms:.1f}ms Det:{det_ms:.1f}ms | GT:{gt_count} Pred:{pred_count}"
        )

    def _schedule_video_ui_update(self, *args):
        if self._video_ui_pending:
            return
        self._video_ui_pending = True
        self.root.after(0, self._update_video_ui_images, *args)

    def _update_video_ui_images(self, *args):
        try:
            self._update_ui_images(*args)
        finally:
            self._video_ui_pending = False

    def _show_image(self, cv_img, tk_label):
        if cv_img is None:
            tk_label.config(text="No Image", image="")
            return
        ww, wh = max(396, tk_label.winfo_width() - 4), max(296, tk_label.winfo_height() - 4)
        h, w = cv_img.shape[:2]
        s = min(ww / max(w, 1), wh / max(h, 1))
        img = cv2.resize(cv_img, (max(1, int(w * s)), max(1, int(h * s))))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        tk_label.config(image=photo, text="")
        self._img_refs[tk_label] = photo

    def _set_no_image_all(self):
        for frame in [self.lbl_rgb, self.lbl_ir, self.lbl_fusion, self.lbl_result]:
            frame.inner_label.config(text="No Image", image="")
            self._img_refs.pop(frame.inner_label, None)

    def open_visible_video(self):
        p = filedialog.askopenfilename(title="选择可见光视频", filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv")])
        if p:
            was_camera = self.stream_mode.startswith("camera_")
            self.stop_video()
            self.stream_mode = "video"
            if was_camera:
                self.ir_video_source = None
            self.video_source = p
            self.video_backend = None
            self.log(f"[GUI] 可见光视频: {p}")
            if self.current_model is not None and self._current_model_channels() < 6:
                self.start_video_inference()
            elif self.ir_video_source is not None:
                self.start_video_inference()

    def open_ir_video(self):
        p = filedialog.askopenfilename(title="选择红外视频", filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv")])
        if p:
            was_camera = self.stream_mode.startswith("camera_")
            self.stop_video()
            self.stream_mode = "video"
            if was_camera:
                self.video_source = None
            self.ir_video_source = p
            self.ir_video_backend = None
            self.log(f"[GUI] 红外视频: {p}")
            if self.video_source is not None:
                self.start_video_inference()

    def open_auto_camera(self):
        if self.camera_scan_worker and self.camera_scan_worker.is_alive():
            self.log("[CAM] 正在扫描摄像头，请稍候")
            return
        self.stop_video()
        self.info_var.set("正在扫描摄像头...")
        self.log("[CAM] 开始自动识别摄像头")
        self.camera_scan_worker = threading.Thread(target=self._scan_cameras, daemon=True)
        self.camera_scan_worker.start()

    def _scan_cameras(self):
        try:
            devices = discover_cameras(validate=True)
            self.root.after(0, self._start_auto_camera, devices)
        except Exception as e:
            self.root.after(0, self._show_camera_scan_error, str(e))

    def _show_camera_scan_error(self, error: str):
        self.info_var.set("摄像头扫描失败")
        self.log(f"[CAM] 摄像头扫描失败: {error}")
        messagebox.showerror("摄像头错误", f"自动识别摄像头失败：\n{error}")

    def _start_auto_camera(self, devices: list[CameraDevice]):
        if not devices:
            self.info_var.set("未检测到可用摄像头")
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
            source_type = "内置前置" if setup.rgb.is_integrated else "外接"
            self.log(
                f"[CAM] 自动使用{source_type}摄像头: {setup.rgb.name}({setup.rgb.index}) | "
                f"3通道模型:{self.model_var.get()}"
            )
        self.start_video_inference()

    def open_rgbt_cameras(self):
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
        self.start_video_inference()

    def toggle_pause_video(self):
        if self.video_running:
            self.video_paused = not self.video_paused
            self.log("[GUI] 视频已暂停" if self.video_paused else "[GUI] 视频继续播放")

    def stop_video(self):
        self.video_running = False
        self.video_paused = False
        self._release_video_io()

    def _release_video_io(self):
        for cap in (self.cap_vis, self.cap_ir):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        self.cap_vis = None
        self.cap_ir = None
        if self.output_writer is not None:
            try:
                self.output_writer.release()
            except Exception:
                pass
        self.output_writer = None

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

        self.stop_video()
        self.video_running = True
        self._video_ui_pending = False
        self.video_worker = threading.Thread(target=self._video_loop, daemon=True)
        self.video_worker.start()


    def _video_loop(self):
        try:
            ch = self._current_model_channels()
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
            if is_camera:
                total_frames = "实时"
            elif use_ir:
                total_frames = int(
                    min(
                        self.cap_vis.get(cv2.CAP_PROP_FRAME_COUNT) or 0,
                        self.cap_ir.get(cv2.CAP_PROP_FRAME_COUNT) or 0,
                    )
                ) or 1
            else:
                total_frames = int(self.cap_vis.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
            delay = max(0.001, 1.0 / (fps * max(0.25, float(self.speed_var.get()))))
            skip_accum = 0.0

            while self.video_running:
                frame_t0 = time.time()
                if self.video_paused:
                    time.sleep(0.05)
                    continue

                rv, fv = self.cap_vis.read()
                if not rv:
                    break
                if use_ir:
                    ri, fi = self.cap_ir.read()
                    if not ri:
                        break
                else:
                    fi = None

                frame_no = int(self.cap_vis.get(cv2.CAP_PROP_POS_FRAMES))
                fuse_t0 = time.time()
                if use_ir:
                    ir_show = fi if fi.ndim == 3 else cv2.cvtColor(fi, cv2.COLOR_GRAY2BGR)
                    ir_preprocessed = self.fusion_engine.preprocess_ir(fi)
                    fused = self.fusion_engine.for_display_with_preprocessed_ir(fv, ir_preprocessed)
                    model_input = self._model_input_with_preprocessed_ir(fv, ir_preprocessed)
                else:
                    ir_show = None
                    fused = fv.copy()
                    model_input = fv
                fuse_ms = (time.time() - fuse_t0) * 1000.0
                res = fused.copy()
                det_ms, pred_count, _det_logs = self._draw_pred(res, model_input)
                self.latest_result_frame = res.copy()

                if self.output_path:
                    if self.output_writer is None:
                        h, w = res.shape[:2]
                        self.output_writer = cv2.VideoWriter(
                            self.output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h),
                        )
                    self.output_writer.write(res)

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
                )
                elapsed = time.time() - frame_t0
                if elapsed < delay:
                    time.sleep(delay - elapsed)
                elif not is_camera:
                    skip_accum += elapsed / delay - 1.0
                    skip_frames = min(8, int(skip_accum))
                    if skip_frames > 0:
                        skip_accum -= skip_frames
                        for _ in range(skip_frames):
                            rv_skip = self.cap_vis.grab()
                            ri_skip = self.cap_ir.grab() if use_ir else True
                            if not rv_skip or not ri_skip:
                                break
        except Exception as e:
            self.log(f"[GUI] 视频推理失败: {e}")
        finally:
            self.video_running = False
            self._release_video_io()

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
            self.output_path = p
            self.log(f"[GUI] 视频结果将保存到: {p}")
        else:
            cv2.imwrite(p, self.latest_result_frame)
            self.log(f"[GUI] 图片结果已保存到: {p}")

    def _on_close(self):
        self.stop_video()
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
