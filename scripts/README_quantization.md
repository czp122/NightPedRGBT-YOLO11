# ONNX模型量化和剪枝工具

本工具用于对YOLO11模型进行ONNX Runtime动态量化和剪枝，无需重新训练数据。

## 功能特性

- **ONNX Runtime动态量化**：对运行时支持的权重执行INT8动态量化
- **结构化稀疏化**：将低重要度卷积滤波器置零，供后续微调和结构压缩实验使用
- **可直接转换**：量化可直接执行；剪枝后若用于精度对比，建议再微调
- ✅ **支持6通道输入**：适配RGBT模型的6通道输入

## 环境依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本量化（不剪枝）

```bash
python scripts/quantize_onnx.py --model best.pt --output quantized_model.onnx
```

### 2. 剪枝+量化

```bash
python scripts/quantize_onnx.py --model best.pt --output pruned_quantized_model.onnx --prune 0.3
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
2. 当前剪枝步骤产生的是稀疏权重，不会自动改变网络通道数；密集推理后端下不保证减小文件或加速
3. YOLO以卷积算子为主，动态量化能否缩小或加速取决于ONNX Runtime版本和CPU指令集，必须实测
4. 量化后的模型使用ONNX Runtime运行，当前GUI仍加载`.pt`模型
5. 工具会从模型首层自动识别3通道或6通道输入

## 示例

### 示例1：仅量化

```bash
python scripts/quantize_onnx.py --model runs/rgbt_yolo11n_cbam11/weights/best.pt --output quantized_best.onnx
```

### 示例2：剪枝30%并量化

```bash
python scripts/quantize_onnx.py --model runs/rgbt_yolo11n_cbam11/weights/best.pt --output pruned_quantized_best.onnx --prune 0.3
```

## 性能验证

脚本会打印FP32与量化后ONNX文件大小。推理速度、内存占用和精度需要在目标电脑及验证集上分别测试，不能把固定倍数作为实验结论。

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
