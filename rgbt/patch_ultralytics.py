from __future__ import annotations

"""Minimal-intrusion RGBT (RGB+IR) dynamic pairing patch for Ultralytics.

This patch keeps the official Ultralytics training pipeline (metrics/logging),
but loads LLVIP visible + infrared pairs on-the-fly to form a 6-channel image:

    6ch = concat([BGR_visible, BGR_infrared_repeat3], axis=2)

Expected LLVIP layout
---------------------
LLVIP/
  visible/
    train/images/*.jpg + train/labels/*.txt
    test/images/*.jpg  + test/labels/*.txt
  infrared/
    train/images/*.jpg
    test/images/*.jpg

You do NOT need to create any fused images on disk.
"""

from pathlib import Path
from typing import Callable, Tuple

import cv2
import numpy as np

from app_utils.preprocess import build_rgbt_for_model


def _to_3ch(im: np.ndarray) -> np.ndarray:
    """Ensure HxWx3."""
    if im is None:
        return im
    if im.ndim == 2:
        return np.repeat(im[..., None], 3, axis=2)
    if im.ndim == 3 and im.shape[2] == 1:
        return np.repeat(im, 3, axis=2)
    if im.ndim == 3 and im.shape[2] >= 3:
        return im[:, :, :3]
    return im


def _infer_ir_path(visible_path: str | Path) -> Path:
    """Map a visible image path to its infrared paired path using LLVIP conventions.

    Rule:
      replace any path component named 'visible' (case-insensitive) with 'infrared'
      keep the rest identical, including train/test/images/filename.
    """
    vp = Path(visible_path)
    parts = list(vp.parts)
    parts = ["infrared" if p.lower() == "visible" else p for p in parts]
    return Path(*parts)


def apply_rgbt_llvip_patch() -> None:
    """Apply monkey patches (safe to call multiple times)."""

    # 1) Patch BaseDataset.load_image to produce 6-channel images
    from ultralytics.data.base import BaseDataset
    from ultralytics.data.base import imread as ul_imread

    if getattr(BaseDataset.load_image, "_rgbt_patched", False):
        return  # already patched

    _orig_load_image: Callable = BaseDataset.load_image

    def load_image_rgbt(self: BaseDataset, i: int, rect_mode: bool = True):  # type: ignore[override]
        # If dataset is not configured for 6ch, fall back to original
        ch = getattr(self, "channels", 3)
        if int(ch) != 6:
            return _orig_load_image(self, i, rect_mode=rect_mode)

        # Mirrors BaseDataset.load_image logic, but replaces `imread(f)` with RGB+IR concat.
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:
            # We intentionally do NOT support cached npy for RGBT to avoid extra disk usage.
            # IMPORTANT: ul_imread expects a string path (not WindowsPath).
            im_v = ul_imread(str(f), flags=self.cv2_flag)  # BGR visible
            if im_v is None:
                raise FileNotFoundError(f"Visible image not found: {f}")

            ir_path = _infer_ir_path(f)
            im_ir = ul_imread(str(ir_path), flags=self.cv2_flag)
            if im_ir is None:
                raise FileNotFoundError(f"Infrared image not found: {ir_path} (from visible: {f})")

            im = build_rgbt_for_model(im_v, im_ir)  # HxWx6
            h0, w0 = im.shape[:2]

            # ----- same resize path as original -----
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w = min(int(np.ceil(w0 * r)), self.imgsz)
                    h = min(int(np.ceil(h0 * r)), self.imgsz)
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            if im.ndim == 2:
                im = im[..., None]

            # Cache behavior (same style as Ultralytics)
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    load_image_rgbt._rgbt_patched = True  # type: ignore[attr-defined]
    BaseDataset.load_image = load_image_rgbt  # type: ignore[assignment]

    # 2) Patch RandomHSV to only apply on the first 3 channels (visible) when img is 6ch
    from ultralytics.data.augment import RandomHSV

    if not getattr(RandomHSV.__call__, "_rgbt_patched", False):
        _orig_call = RandomHSV.__call__

        def call_rgbt(self: RandomHSV, labels):
            im = labels.get("img")
            if isinstance(im, np.ndarray) and im.ndim == 3 and im.shape[2] == 6:
                rgb = im[:, :, :3].copy()
                labels_rgb = dict(labels)
                labels_rgb["img"] = rgb
                labels_rgb = _orig_call(self, labels_rgb)
                im[:, :, :3] = labels_rgb["img"]
                labels["img"] = im
                return labels
            return _orig_call(self, labels)

        call_rgbt._rgbt_patched = True  # type: ignore[attr-defined]
        RandomHSV.__call__ = call_rgbt  # type: ignore[assignment]


def llvip_sanity_check(llvip_root: str | Path) -> Tuple[Path, Path]:
    """Quick check for the LLVIP directory layout. Returns (visible_root, infrared_root)."""
    root = Path(llvip_root)
    v = root / "visible"
    ir = root / "infrared"
    if not v.exists() or not ir.exists():
        raise FileNotFoundError(f"LLVIP root must contain visible/ and infrared/. Got: {root}")
    return v, ir
