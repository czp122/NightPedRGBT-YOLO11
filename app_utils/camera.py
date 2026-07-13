from __future__ import annotations

import os
from dataclasses import dataclass

import cv2

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:  # The OpenCV probing fallback still works in source deployments.
    enumerate_cameras = None


INTEGRATED_HINTS = (
    "integrated",
    "internal",
    "built-in",
    "builtin",
    "front camera",
    "facetime",
    "内置",
    "前置",
)
IR_HINTS = (
    "infrared",
    "ir camera",
    "ir webcam",
    "thermal",
    "thermographic",
    "flir",
    "seek thermal",
    "热成像",
    "红外",
)


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    backend: int
    path: str = ""

    @property
    def is_integrated(self) -> bool:
        name = self.name.lower()
        return any(hint in name for hint in INTEGRATED_HINTS)

    @property
    def is_ir(self) -> bool:
        name = self.name.lower()
        return any(hint in name for hint in IR_HINTS)

    @property
    def kind(self) -> str:
        if self.is_ir:
            return "IR"
        return "内置RGB" if self.is_integrated else "外接RGB"


@dataclass(frozen=True)
class CameraSetup:
    mode: str
    rgb: CameraDevice
    ir: CameraDevice | None = None


def _open_camera(device: CameraDevice):
    if os.name == "nt":
        cap = cv2.VideoCapture(device.index, device.backend or cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(device.index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _can_read(device: CameraDevice) -> bool:
    cap = _open_camera(device)
    try:
        if not cap.isOpened():
            return False
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return True
        return False
    finally:
        cap.release()


def _opencv_probe(max_index: int) -> list[CameraDevice]:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    devices = []
    misses = 0
    for index in range(max_index + 1):
        device = CameraDevice(index, f"Camera {index}", backend)
        if _can_read(device):
            devices.append(device)
            misses = 0
        else:
            misses += 1
            if devices and misses >= 2:
                break
    return devices


def discover_cameras(validate: bool = True, max_index: int = 8) -> list[CameraDevice]:
    """Return working cameras in the same index order used by OpenCV."""
    devices: list[CameraDevice] = []
    if enumerate_cameras is not None and os.name == "nt":
        for info in enumerate_cameras(cv2.CAP_DSHOW):
            devices.append(
                CameraDevice(
                    index=int(info.index),
                    name=str(info.name),
                    backend=int(info.backend),
                    path=str(info.path),
                )
            )
    else:
        devices = _opencv_probe(max_index)

    if validate:
        devices = [device for device in devices if _can_read(device)]
    return devices


def choose_auto_camera_setup(devices: list[CameraDevice]) -> CameraSetup | None:
    """Prefer an external IR+RGB pair, then external RGB, then integrated RGB."""
    if not devices:
        return None

    ir_devices = [device for device in devices if device.is_ir]
    external_rgb = [device for device in devices if not device.is_ir and not device.is_integrated]
    integrated_rgb = [device for device in devices if not device.is_ir and device.is_integrated]
    other_rgb = [device for device in devices if not device.is_ir]

    if ir_devices:
        rgb_candidates = external_rgb + integrated_rgb + other_rgb
        for ir_device in ir_devices:
            rgb_device = next((d for d in rgb_candidates if d.index != ir_device.index), None)
            if rgb_device is not None:
                return CameraSetup("rgbt", rgb_device, ir_device)

    if external_rgb:
        return CameraSetup("rgb", external_rgb[0])
    if integrated_rgb:
        return CameraSetup("rgb", integrated_rgb[0])
    if other_rgb:
        return CameraSetup("rgb", other_rgb[0])
    return None
