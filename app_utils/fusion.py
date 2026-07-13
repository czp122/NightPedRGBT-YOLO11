from __future__ import annotations

import cv2
import numpy as np

from app_utils.preprocess import build_rgbt_for_model, preprocess_ir_for_model


def _ensure_3ch(im: np.ndarray) -> np.ndarray:
    """确保图片是 3 通道 (H, W, 3)。"""
    if im is None:
        return None
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.ndim == 3 and im.shape[2] == 1:
        return np.repeat(im, 3, axis=2)
    if im.ndim == 3 and im.shape[2] >= 3:
        return im[:, :, :3]
    return im


def _safe_clahe_bgr(im: np.ndarray, clahe) -> np.ndarray:
    try:
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return im


class FusionEngine:
    def __init__(self, use_clahe: bool = True, use_gaussian: bool = True):
        self.use_clahe = bool(use_clahe)
        self.use_gaussian = bool(use_gaussian)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if self.use_clahe else None
        print("[FusionEngine] Initialized (adaptive RGBT fusion enabled)")

    def fuse(self, rgb, ir):
        """兼容旧代码的接口，默认返回用于显示的 3 通道图像。"""
        return self.for_display(rgb, ir)

    def preprocess_ir(self, ir_bgr: np.ndarray) -> np.ndarray:
        """# 优化：红外预处理增加高斯去噪 + CLAHE，对外接口保持不变。"""
        return preprocess_ir_for_model(ir_bgr)

    @staticmethod
    def _align_modalities(rgb: np.ndarray, ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """# 优化：通过边缘重心做轻量级特征对齐，避免跨模态明显错位。"""
        if rgb is None or ir is None:
            return rgb, ir

        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        rgb_edge = cv2.Canny(rgb_gray, 50, 150)
        ir_edge = cv2.Canny(ir_gray, 50, 150)

        rgb_pts = np.column_stack(np.where(rgb_edge > 0))
        ir_pts = np.column_stack(np.where(ir_edge > 0))
        if len(rgb_pts) < 16 or len(ir_pts) < 16:
            return rgb, ir

        rgb_center = rgb_pts.mean(axis=0)
        ir_center = ir_pts.mean(axis=0)
        dy, dx = rgb_center - ir_center
        dx = int(np.clip(np.round(dx), -8, 8))
        dy = int(np.clip(np.round(dy), -8, 8))

        if dx == 0 and dy == 0:
            return rgb, ir

        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_ir = cv2.warpAffine(ir, mat, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return rgb, aligned_ir

    @staticmethod
    def _adaptive_weights(rgb: np.ndarray, ir: np.ndarray) -> tuple[float, float]:
        """# 优化：依据亮度/梯度强度自适应调整融合权重。"""
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        rgb_grad = cv2.Laplacian(rgb_gray, cv2.CV_32F).var()
        ir_grad = cv2.Laplacian(ir_gray, cv2.CV_32F).var()
        rgb_mean = float(np.mean(rgb_gray)) + 1e-6
        ir_mean = float(np.mean(ir_gray)) + 1e-6

        rgb_score = 0.6 * rgb_grad + 0.4 * rgb_mean
        ir_score = 0.6 * ir_grad + 0.4 * ir_mean
        total = rgb_score + ir_score + 1e-6

        w_rgb = float(np.clip(rgb_score / total, 0.35, 0.75))
        w_ir = 1.0 - w_rgb
        return w_rgb, w_ir

    def for_model(self, rgb_bgr: np.ndarray, ir_bgr: np.ndarray) -> np.ndarray:
        """
        生成 6 通道数据供 YOLO 推理。
        返回: (H, W, 6)
        """
        return build_rgbt_for_model(rgb_bgr, ir_bgr)

    def for_model_with_preprocessed_ir(self, rgb_bgr: np.ndarray, ir_bgr: np.ndarray) -> np.ndarray:
        """Use an already-preprocessed IR image to avoid duplicate denoising/CLAHE work."""
        rgb = _ensure_3ch(rgb_bgr)
        ir = _ensure_3ch(ir_bgr)
        if rgb is None or ir is None:
            raise ValueError("rgb/ir input must not be None")
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.concatenate([rgb, ir], axis=2)

    def for_display(self, rgb_bgr: np.ndarray, ir_bgr: np.ndarray) -> np.ndarray:
        """
        生成 3 通道伪彩融合图供人眼观看。
        返回: (H, W, 3)
        """
        rgb = _ensure_3ch(rgb_bgr)
        ir = self.preprocess_ir(ir_bgr)
        return self.for_display_with_preprocessed_ir(rgb, ir)

    def for_display_with_preprocessed_ir(self, rgb_bgr: np.ndarray, ir_bgr: np.ndarray) -> np.ndarray:
        """Build the display fusion image using an already-preprocessed IR image."""
        rgb = _ensure_3ch(rgb_bgr)
        ir = _ensure_3ch(ir_bgr)
        if rgb is None or ir is None:
            raise ValueError("rgb/ir input must not be None")

        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        rgb_disp = _safe_clahe_bgr(rgb.copy(), self.clahe) if self.clahe is not None else rgb.copy()
        rgb_disp, ir = self._align_modalities(rgb_disp, ir)

        ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        ir_color = cv2.applyColorMap(ir_gray, cv2.COLORMAP_JET)
        w_rgb, w_ir = self._adaptive_weights(rgb_disp, ir)

        out = cv2.addWeighted(rgb_disp, w_rgb, ir_color, w_ir, 0.0)
        return out
