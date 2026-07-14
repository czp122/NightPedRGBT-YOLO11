from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def prune_model(model: nn.Module, prune_amount: float = 0.3):
    """Apply structured filter sparsification without changing the dense graph shape."""
    print(f"开始剪枝，剪枝比例: {prune_amount}")
    prune_amount = float(prune_amount)
    if not 0.0 <= prune_amount <= 0.9:
        raise ValueError(f"剪枝比例必须在0到0.9之间，当前为: {prune_amount}")

    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1:
            continue
        if module.out_channels < 16:
            continue
        if any(skip in name.lower() for skip in ("detect", "dfl", "proto", "head")):
            continue
        modules.append((module, "weight"))

    if not modules:
        print("未找到可剪枝卷积层，跳过")
        return model

    for module, param_name in modules:
        prune.ln_structured(module, name=param_name, amount=prune_amount, n=2, dim=0)
        prune.remove(module, param_name)

    print(f"剪枝完成，处理卷积层数量: {len(modules)}")
    return model


def _resolve_device(device: str) -> torch.device:
    requested = (device or "cpu").strip().lower()
    if requested == "auto":
        requested = "0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda:"):
        requested = requested.split(":", 1)[1]
    if not requested.isdigit():
        raise ValueError(f"不支持的设备参数: {device}，请使用cpu、auto、0或cuda:0")
    index = int(requested)
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(f"请求CUDA:{index}，但该GPU当前不可用")
    return torch.device(f"cuda:{index}")


class _PredictionOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images)
        return output[0] if isinstance(output, (tuple, list)) else output


def export_to_onnx(model: nn.Module, output_path, imgsz: int = 640, device: str = "cpu"):
    print(f"导出模型到ONNX格式: {output_path}")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not 32 <= int(imgsz) <= 2048:
        raise ValueError(f"imgsz必须在32到2048之间，当前为: {imgsz}")
    model = model.eval()
    dev = _resolve_device(device)
    model.to(dev)
    channels = int(getattr(model, "yaml", {}).get("channels", 0) or 0)
    if channels <= 0:
        first_conv = next((module for module in model.modules() if isinstance(module, nn.Conv2d)), None)
        channels = int(first_conv.in_channels) if first_conv is not None else 3

    input_tensor = torch.randn(1, channels, imgsz, imgsz, device=dev)
    export_model = _PredictionOnly(model).eval()
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            input_tensor,
            str(output_path),
            opset_version=17,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "output0": {0: "batch", 2: "anchors"},
            },
            do_constant_folding=True,
        )
    print(f"ONNX导出完成，输入通道: {channels}")
    return str(output_path)


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    """# 优化：采用 PTQ 动态量化，避免对卷积路径过度量化导致精度暴跌。"""
    print(f"开始PTQ动态量化ONNX模型: {onnx_model_path}")
    model_input = Path(onnx_model_path).expanduser().resolve()
    model_output = Path(quantized_model_path).expanduser().resolve()
    if not model_input.is_file():
        raise FileNotFoundError(f"ONNX模型不存在: {model_input}")
    if model_input == model_output:
        raise ValueError("量化输入和输出路径不能相同")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(model_input),
        model_output=str(model_output),
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=["/model.23", "/model.24", "Detect"],
    )
    before_mb = model_input.stat().st_size / (1024**2)
    after_mb = model_output.stat().st_size / (1024**2)
    print(f"量化完成: {model_output} | {before_mb:.1f} MB -> {after_mb:.1f} MB")
    return str(model_output)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ONNX模型量化和剪枝工具")
    parser.add_argument("--model", type=str, default="best.pt", help="输入模型路径")
    parser.add_argument("--output", type=str, default="quantized_model.onnx", help="输出量化模型路径")
    parser.add_argument("--prune", type=float, default=0.0, help="剪枝比例，0表示不剪枝")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小")
    parser.add_argument("--device", type=str, default="cpu", help='设备，如 "cpu" 或 "cuda:0"')
    args = parser.parse_args()

    if not 0.0 <= args.prune <= 0.9:
        raise ValueError("--prune必须在0到0.9之间")
    if not 32 <= args.imgsz <= 2048:
        raise ValueError("--imgsz必须在32到2048之间")

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    print(f"加载模型: {model_path}")
    yolo_model = YOLO(str(model_path))
    model = yolo_model.model

    if args.prune > 0:
        model = prune_model(model, args.prune)

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.resolve()
    onnx_path = output_path.with_name(f"{output_path.stem}_fp32.onnx")
    export_to_onnx(model, onnx_path, args.imgsz, args.device)
    quantize_onnx_model(onnx_path, output_path)
    print("\n所有操作完成！")


if __name__ == "__main__":
    main()
