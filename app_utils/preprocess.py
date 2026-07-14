from __future__ import annotations

import cv2
import numpy as np


def to_uint8(im: np.ndarray | None) -> np.ndarray | None:
    """Convert camera or TIFF data to uint8 without wrapping 16-bit values."""
    if im is None or im.dtype == np.uint8:
        return im
    array = np.asarray(im)
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255

    work = array.astype(np.float32, copy=False)
    finite = work[np.isfinite(work)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum >= 0.0 and maximum <= 1.0:
        scaled = work * 255.0
        return np.nan_to_num(np.clip(scaled, 0, 255), nan=0.0).astype(np.uint8)
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (work - minimum) * (255.0 / (maximum - minimum))
    return np.nan_to_num(np.clip(scaled, 0, 255), nan=0.0).astype(np.uint8)


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


def preprocess_ir_for_model(
    ir_bgr: np.ndarray | None,
    realtime: bool = False,
    realtime_max_side: int = 320,
) -> np.ndarray | None:
    """Apply grayscale, NLM denoising and CLAHE; realtime mode reduces video latency."""
    ir = ensure_3ch(ir_bgr)
    if ir is None:
        return None

    gray = to_uint8(cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY))
    work = gray
    if realtime and max(gray.shape) > realtime_max_side:
        scale = realtime_max_side / max(gray.shape)
        work = cv2.resize(
            gray,
            (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    template_window = 5 if realtime else 7
    search_window = 15 if realtime else 21
    denoised = cv2.fastNlMeansDenoising(
        work,
        None,
        h=5,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    if enhanced.shape != gray.shape:
        enhanced = cv2.resize(
            enhanced,
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
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
