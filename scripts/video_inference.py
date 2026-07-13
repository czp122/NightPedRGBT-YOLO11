from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, List
import argparse

from ultralytics import YOLO
from app_utils.fusion import FusionEngine


def _to_3ch(im: np.ndarray) -> np.ndarray:
    if im is None:
        return im
    if im.ndim == 2:
        return np.repeat(im[..., None], 3, axis=2)
    if im.ndim == 3 and im.shape[2] == 1:
        return np.repeat(im, 3, axis=2)
    if im.ndim == 3 and im.shape[2] >= 3:
        return im[:, :, :3]
    return im


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    fusion: FusionEngine,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
) -> Tuple[np.ndarray, List[dict]]:
    """
    处理单帧图像
    
    Args:
        frame: 输入帧（BGR格式）
        model: YOLO模型
        imgsz: 推理图像大小
        conf: 置信度阈值
        iou: IOU阈值
    
    Returns:
        带检测框的图像，检测结果列表
    """
    # 复制原始帧用于绘制
    vis = frame.copy()
    
    # 为了模拟RGBT输入，我们将同一帧复制为红外通道
    # 实际应用中，应该使用真实的红外图像
    ir_frame = frame.copy()
    
    # 转换为3通道
    frame = _to_3ch(frame)
    ir_frame = _to_3ch(ir_frame)
    
    # 确保尺寸一致
    if ir_frame.shape[:2] != frame.shape[:2]:
        ir_frame = cv2.resize(ir_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # 创建6通道输入
    im6 = fusion.for_model(frame, ir_frame)
    
    # 统一走 Ultralytics predict 路径，避免与 GUI/验证后处理不一致
    results = model.predict(
        im6,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    det_list = []
    boxes = results[0].boxes if results else None

    if boxes is not None:
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            score = float(b.conf[0])
            cls = int(b.cls[0])
            x1, y1, x2, y2 = xyxy.tolist()
            det_list.append({"xyxy": (x1, y1, x2, y2), "conf": score, "cls": cls})

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"person {score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    
    return vis, det_list


def process_video(
    input_video: str,
    output_video: str,
    model_weights: str,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    show: bool = False,
):
    """
    处理视频文件
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        model_weights: 模型权重路径
        device: 设备（"cpu"或"cuda:0"）
        imgsz: 推理图像大小
        conf: 置信度阈值
        iou: IOU阈值
        show: 是否显示视频
    """
    # 加载模型
    print(f"加载模型: {model_weights}")
    model = YOLO(model_weights)
    fusion = FusionEngine()
    if device != "cpu":
        model.to(device)
    
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {input_video}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"开始处理视频: {input_video}")
    print(f"视频信息: {width}x{height}, {fps} FPS")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame, dets = process_frame(frame, model, fusion, imgsz, conf, iou)
        
        # 写入输出视频
        out.write(processed_frame)
        
        # 显示视频
        if show:
            cv2.imshow('Night Pedestrian Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理帧: {frame_count}")
    
    # 释放资源
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f"视频处理完成，保存到: {output_video}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="视频推理脚本")
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('--model', type=str, default='best.pt', help='模型权重路径')
    parser.add_argument('--device', type=str, default='cpu', help='设备（"cpu"或"cuda:0"）')
    parser.add_argument('--imgsz', type=int, default=640, help='推理图像大小')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--show', action='store_true', help='是否显示视频')
    
    args = parser.parse_args()
    
    process_video(
        input_video=args.input,
        output_video=args.output,
        model_weights=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
    )


if __name__ == '__main__':
    main()
