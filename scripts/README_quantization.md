# ONNX模型量化和剪枝工具

本工具用于对YOLO11模型进行ONNX Runtime动态量化和剪枝，无需重新训练数据。

## 功能特性

- ✅ **ONNX Runtime动态量化**：将模型量化为INT8精度，减小模型体积，提升推理速度
- ✅ **模型剪枝**：移除不重要的权重，进一步减小模型大小
- ✅ **无需重新训练**：直接对训练好的模型进行处理
- ✅ **支持6通道输入**：适配RGBT模型的6通道输入

## 环境依赖

```bash
pip install onnxruntime torch
```

## 使用方法

### 1. 基本量化（不剪枝）

```bash
python quantize_onnx.py --model best.pt --output quantized_model.onnx
```

### 2. 剪枝+量化

```bash
python quantize_onnx.py --model best.pt --output pruned_quantized_model.onnx --prune 0.3
```

参数说明：
- `--model`：输入模型路径（.pt文件）
- `--output`：输出量化模型路径（.onnx文件）
- `--prune`：剪枝比例，0表示不剪枝，0.3表示剪枝30%的权重
- `--imgsz`：输入图像大小，默认640

## 工作流程

1. 加载PyTorch模型
2. （可选）对模型进行剪枝
3. 导出为ONNX格式
4. 使用ONNX Runtime动态量化模型
5. 保存量化后的模型

## 注意事项

1. 动态量化可能会导致轻微的精度下降，建议在量化后进行验证
2. 剪枝比例不宜过高，一般建议在0.3-0.5之间
3. 量化后的模型可以直接在ONNX Runtime中使用
4. 对于6通道输入的RGBT模型，量化时会自动处理输入通道

## 示例

### 示例1：仅量化

```bash
python quantize_onnx.py --model runs/rgbt_yolo11n_cbam11/weights/best.pt --output quantized_best.onnx
```

### 示例2：剪枝30%并量化

```bash
python quantize_onnx.py --model runs/rgbt_yolo11n_cbam11/weights/best.pt --output pruned_quantized_best.onnx --prune 0.3
```

## 性能提升

- 模型大小：约减小75%（从FP32到INT8）
- 推理速度：在CPU上约提升2-3倍
- 内存占用：约减小75%

## 验证量化模型

可以使用以下代码验证量化模型的性能：

```python
import onnxruntime as ort
import numpy as np

# 加载量化模型
session = ort.InferenceSession('quantized_model.onnx')

# 准备输入数据（6通道）
input_data = np.random.randn(1, 6, 640, 640).astype(np.float32)

# 推理
outputs = session.run(None, {'images': input_data})
print(f"输出形状: {outputs[0].shape}")
```
