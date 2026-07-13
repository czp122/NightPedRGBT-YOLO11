from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prune_model(model: nn.Module, prune_amount: float = 0.3):
    """# 优化：优先保留关键层，采用更稳健的结构化剪枝。"""
    print(f"开始剪枝，剪枝比例: {prune_amount}")
    prune_amount = float(max(0.0, min(0.9, prune_amount)))

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


def export_to_onnx(model: nn.Module, output_path, imgsz: int = 640, device: str = "cpu"):
    print(f"导出模型到ONNX格式: {output_path}")
    output_path = str(output_path)
    model = model.eval()
    dev = torch.device(device if device != "0" else "cuda:0") if device != "cpu" and torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    input_tensor = torch.randn(1, 6, imgsz, imgsz, device=dev)
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tensor,
            output_path,
            opset_version=12,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}, "output0": {0: "batch"}},
            do_constant_folding=True,
        )
    print("ONNX导出完成")
    return output_path


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    """# 优化：采用 PTQ 动态量化，避免对卷积路径过度量化导致精度暴跌。"""
    print(f"开始PTQ动态量化ONNX模型: {onnx_model_path}")
    quantize_dynamic(
        model_input=str(onnx_model_path),
        model_output=str(quantized_model_path),
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=["/model.23", "/model.24", "Detect"],
    )
    print(f"量化完成，模型保存到: {quantized_model_path}")
    return quantized_model_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ONNX模型量化和剪枝工具")
    parser.add_argument("--model", type=str, default="best.pt", help="输入模型路径")
    parser.add_argument("--output", type=str, default="quantized_model.onnx", help="输出量化模型路径")
    parser.add_argument("--prune", type=float, default=0.0, help="剪枝比例，0表示不剪枝")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小")
    parser.add_argument("--device", type=str, default="cpu", help='设备，如 "cpu" 或 "cuda:0"')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    print(f"加载模型: {model_path}")
    yolo_model = YOLO(str(model_path))
    model = yolo_model.model

    if args.prune > 0:
        model = prune_model(model, args.prune)

    onnx_path = model_path.with_suffix(".onnx")
    export_to_onnx(model, onnx_path, args.imgsz, args.device)
    quantize_onnx_model(onnx_path, args.output)
    print("\n所有操作完成！")


if __name__ == "__main__":
    main()
