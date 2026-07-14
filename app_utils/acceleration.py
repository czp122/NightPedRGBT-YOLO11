from __future__ import annotations

import os
from dataclasses import dataclass

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import cv2
import torch


@dataclass(frozen=True)
class ComputeDevice:
    predict_arg: str
    label: str
    is_cuda: bool
    use_half: bool
    fallback_reason: str = ""


def configure_compute_runtime() -> int:
    """Configure conservative thread counts and GPU math backends once at startup."""
    cpu_count = max(1, os.cpu_count() or 1)
    cpu_threads = max(1, min(8, cpu_count - 1 if cpu_count > 2 else cpu_count))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    except RuntimeError:
        pass

    # Fusion preprocessing and PyTorch otherwise compete for every CPU core.
    cv2.setNumThreads(max(1, min(4, cpu_threads // 2 or 1)))

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    return cpu_threads


def available_device_options() -> list[str]:
    options = ["auto", "cpu"]
    if torch.cuda.is_available():
        options.extend(str(index) for index in range(torch.cuda.device_count()))
    return options


def resolve_compute_device(selection: str, cpu_threads: int) -> ComputeDevice:
    requested = (selection or "auto").strip().lower()
    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0

    if requested == "auto":
        if cuda_available:
            requested = "0"
        else:
            return ComputeDevice(
                "cpu",
                f"CPU ({cpu_threads} threads)",
                False,
                False,
                "未检测到可用的CUDA设备，已自动使用CPU",
            )

    if requested.startswith("cuda:"):
        requested = requested.split(":", 1)[1]

    if requested.isdigit():
        index = int(requested)
        if cuda_available and index < torch.cuda.device_count():
            name = torch.cuda.get_device_name(index)
            return ComputeDevice(str(index), f"CUDA:{index} {name} (FP16)", True, True)
        return ComputeDevice(
            "cpu",
            f"CPU ({cpu_threads} threads)",
            False,
            False,
            f"CUDA:{index}不可用，已回退到CPU",
        )

    return ComputeDevice("cpu", f"CPU ({cpu_threads} threads)", False, False)


def synchronize_if_cuda(device: ComputeDevice):
    if device.is_cuda:
        torch.cuda.synchronize(int(device.predict_arg))


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_runtime_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "cuda",
            "cudnn",
            "out of memory",
            "device-side assert",
            "no kernel image",
        )
    )
