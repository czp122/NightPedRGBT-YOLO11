from __future__ import annotations

import cv2
import numpy as np


def ensure_3ch(im: np.ndarray | None) -> np.ndarray | None:
    """Return an HxWx3 image while preserving BGR channel order."""
    if im is None:
        return None
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.ndim == 3 and im.shape[2] == 1:
        return np.repeat(im, 3, axis=2)
    if im.ndim == 3 and im.shape[2] >= 3:
        return im[:, :, :3]
    return im


def preprocess_ir_for_model(ir_bgr: np.ndarray | None) -> np.ndarray | None:
    """Apply the thesis IR preprocessing path: grayscale, NLM denoise, CLAHE, then 3ch."""
    ir = ensure_3ch(ir_bgr)
    if ir is None:
        return None

    gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def build_rgbt_for_model(visible_bgr: np.ndarray, ir_bgr: np.ndarray) -> np.ndarray:
    """Build the 6-channel model input used consistently by training and inference."""
    visible = ensure_3ch(visible_bgr)
    ir = preprocess_ir_for_model(ir_bgr)
    if visible is None or ir is None:
        raise ValueError("visible/ir input must not be None")

    if ir.shape[:2] != visible.shape[:2]:
        ir = cv2.resize(ir, (visible.shape[1], visible.shape[0]), interpolation=cv2.INTER_LINEAR)

    return np.concatenate([visible, ir], axis=2)
