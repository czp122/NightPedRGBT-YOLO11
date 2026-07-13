# 视频推理功能

本目录提供了两个视频推理脚本，用于使用训练好的RGBT模型进行夜间行人检测。

## 功能特性

- ✅ **单视频输入**：使用单个视频文件进行推理（模拟RGBT输入）
- ✅ **双视频输入**：使用可见光和红外两个视频文件进行真实RGBT推理
- ✅ **实时显示**：支持实时显示检测结果
- ✅ **视频保存**：将检测结果保存为新的视频文件
- ✅ **参数可调**：支持调整置信度阈值、IOU阈值等参数

## 环境依赖

```bash
pip install opencv-python torch
```

## 脚本说明

### 1. 单视频输入脚本 (`video_inference.py`)

适用于只有单个视频文件的情况，脚本会将同一视频作为可见光和红外输入（模拟RGBT）。

**使用方法**：

```bash
python video_inference.py --input input_video.mp4 --output output_video.mp4 --model best.pt --device cpu
```

**参数说明**：
- `--input`：输入视频路径
- `--output`：输出视频路径
- `--model`：模型权重路径（默认：best.pt）
- `--device`：设备（"cpu"或"cuda:0"，默认：cpu）
- `--imgsz`：推理图像大小（默认：640）
- `--conf`：置信度阈值（默认：0.25）
- `--iou`：IOU阈值（默认：0.45）
- `--show`：是否显示视频（默认：False）

### 2. 双视频输入脚本 (`video_inference_rgbt.py`)

适用于有可见光和红外两个视频文件的情况，实现真实的RGBT推理。

**使用方法**：

```bash
python video_inference_rgbt.py --visible visible_video.mp4 --ir infrared_video.mp4 --output output_video.mp4 --model best.pt --device cpu
```

**参数说明**：
- `--visible`：可见光视频路径
- `--ir`：红外视频路径
- `--output`：输出视频路径
- `--model`：模型权重路径（默认：best.pt）
- `--device`：设备（"cpu"或"cuda:0"，默认：cpu）
- `--imgsz`：推理图像大小（默认：640）
- `--conf`：置信度阈值（默认：0.25）
- `--iou`：IOU阈值（默认：0.45）
- `--show`：是否显示视频（默认：False）

## 示例

### 示例1：使用单视频输入（模拟RGBT）

```bash
python video_inference.py --input night_video.mp4 --output detected_night_video.mp4 --model runs/rgbt_yolo11n_cbam11/weights/best.pt --device cuda:0 --show
```

### 示例2：使用双视频输入（真实RGBT）

```bash
python video_inference_rgbt.py --visible visible_video.mp4 --ir infrared_video.mp4 --output detected_rgbt_video.mp4 --model runs/rgbt_yolo11n_cbam11/weights/best.pt --device cuda:0
```

## 注意事项

1. 视频文件格式：支持常见的视频格式，如MP4、AVI等
2. 视频同步：使用双视频输入时，确保可见光和红外视频是同步的
3. 性能优化：对于较长的视频，建议使用GPU加速（`--device cuda:0`）
4. 输出质量：输出视频的分辨率与输入视频相同
5. 检测精度：可以通过调整`--conf`和`--iou`参数来平衡检测精度和速度

## 性能参考

- **CPU**：Intel i7-10700K，处理720p视频约10-15 FPS
- **GPU**：NVIDIA RTX 3080，处理720p视频约30-60 FPS

## 故障排除

- **视频无法打开**：检查视频路径是否正确，视频文件是否损坏
- **推理速度慢**：尝试使用GPU加速，或减小`--imgsz`值
- **检测结果不准确**：调整`--conf`和`--iou`参数，或使用更好的模型权重
- **内存不足**：减小`--imgsz`值，或处理较短的视频片段
