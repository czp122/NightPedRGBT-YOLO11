from __future__ import annotations

import os
from pathlib import Path

import cv2


class DatasetManager:
    """
    LLVIP 目录结构：
    LLVIP/
      visible/
        train/images/*.jpg
        train/labels/*.txt
        test/images/*.jpg
        test/labels/*.txt
        val/images/*.jpg
        val/labels/*.txt
      infrared/
        train/images/*.jpg
        test/images/*.jpg
        val/images/*.jpg
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, llvip_root, logger=None):
        self.llvip_root = Path(llvip_root)
        self.logger = logger or (lambda x: None)
        self.current_split = None
        self.current_data = []
        self._check_layout()

    def _check_layout(self):
        v = self.llvip_root / "visible"
        ir = self.llvip_root / "infrared"
        self.logger(f"[DatasetManager] root={self.llvip_root}")
        self.logger(f"[DatasetManager] visible exists={v.exists()} infrared exists={ir.exists()}")
        if not v.exists() or not ir.exists():
            raise FileNotFoundError(f"LLVIP root must contain visible/ and infrared/. Got: {self.llvip_root}")

    def _resolve_split(self, split: str) -> str:
        split = str(split).lower().strip()
        if split == "valid":
            split = "val"
        if split not in ("train", "test", "val"):
            self.logger(f"[DatasetManager] unknown split={split}, fallback to test")
            split = "test"

        if (self.llvip_root / "visible" / split).exists() and (self.llvip_root / "infrared" / split).exists():
            return split

        # 优化：兼容无 val 时自动回退 test，避免路径报错。
        if split == "val":
            self.logger("[DatasetManager] val split not found, fallback to test")
            return "test"
        return split

    def load_split(self, split: str) -> int:
        split = self._resolve_split(split)
        self.current_split = split
        self.current_data.clear()

        v_img_dir = self.llvip_root / "visible" / split / "images"
        v_lab_dir = self.llvip_root / "visible" / split / "labels"
        ir_img_dir = self.llvip_root / "infrared" / split / "images"

        self.logger(f"[DatasetManager] load_split={split}")
        self.logger(f"[DatasetManager] v_img_dir={v_img_dir}")
        self.logger(f"[DatasetManager] v_lab_dir={v_lab_dir}")
        self.logger(f"[DatasetManager] ir_img_dir={ir_img_dir}")

        if not v_img_dir.exists():
            self.logger(f"[DatasetManager][ERROR] visible images folder not found: {v_img_dir}")
            return 0
        if not ir_img_dir.exists():
            self.logger(f"[DatasetManager][ERROR] infrared images folder not found: {ir_img_dir}")
            return 0

        vis_files = [p for p in sorted(v_img_dir.iterdir()) if p.is_file() and p.suffix.lower() in self.IMG_EXTS]
        self.logger(f"[DatasetManager] visible images found: {len(vis_files)}")
        if not vis_files:
            return 0

        missing_ir = 0
        missing_label = 0
        for vp in vis_files:
            irp = ir_img_dir / vp.name
            if not irp.exists():
                alt = self._find_by_stem(ir_img_dir, vp.stem)
                if alt is None:
                    missing_ir += 1
                    continue
                irp = alt

            lp = v_lab_dir / f"{vp.stem}.txt"
            if not lp.exists():
                missing_label += 1
                lp = None

            self.current_data.append({"rgb": vp, "ir": irp, "label": lp, "name": vp.name})

        self.logger(
            f"[DatasetManager] paired: {len(self.current_data)} | missing_ir={missing_ir} | missing_label={missing_label}"
        )
        if not self.current_data:
            self.logger("[DatasetManager][HINT] 0 paired samples. Check IR filenames and stems.")
        return len(self.current_data)

    def _find_by_stem(self, folder: Path, stem: str):
        for ext in self.IMG_EXTS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS and p.stem == stem:
                return p
        return None

    def __len__(self):
        return len(self.current_data)

    def get_item(self, idx: int):
        if idx < 0 or idx >= len(self.current_data):
            return None

        item = self.current_data[idx]
        rgb_path, ir_path, label_path, name = item["rgb"], item["ir"], item["label"], item["name"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)

        if rgb is None or ir is None:
            self.logger(f"[DatasetManager][WARN] failed to read: rgb={rgb_path} ir={ir_path}")
            return None

        gt_boxes = []
        if label_path is not None and label_path.exists():
            try:
                gt_boxes = self._read_yolo_label(label_path)
            except Exception as e:
                self.logger(f"[DatasetManager][WARN] read label failed: {label_path} err={e}")
                gt_boxes = []

        return rgb, ir, gt_boxes, name

    def _read_yolo_label(self, txt_path: Path):
        boxes = []
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return boxes

        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue
            if len(nums) == 5:
                cls, cx, cy, w, h = nums
                boxes.append((cx, cy, w, h, int(cls)))
            elif len(nums) == 4:
                cx, cy, w, h = nums
                boxes.append((cx, cy, w, h))
        return boxes
