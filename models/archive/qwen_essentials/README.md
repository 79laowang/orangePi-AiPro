# Qwen2.5 to Ascend 310B OM Model Conversion Guide

本文档提供将 Qwen2.5 模型转换为 Ascend 310B NPU 可用 .om 格式的完整指南。

**[点击查看完整指南 (包含故障排除)](./QWEN_TO_ASCEND_GUIDE.md)**

## 目录结构

```
qwen_onnx_export/
├── export_qwen_to_onnx.py    # PyTorch模型转ONNX脚本
├── convert_onnx_to_om.sh     # ONNX转OM脚本
├── acl_inference_qwen.py     # ACL推理脚本
├── requirements.txt          # Python依赖
├── README.md                 # 本文档
└── QWEN_TO_ASCEND_GUIDE.md   # 完整指南（含故障排除）
```

## 第一步：环境准备

### 1. 安装 Python 依赖

```bash
cd /home/HwHiAiUser/ai-works/orangePi-AiPro/models/qwen_onnx_export

pip3 install -r requirements.txt
```

### 2. 下载 Qwen2.5-3B-Instruct 模型

```bash
# 使用 HuggingFace Hub 下载
pip3 install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./qwen2.5-3b-instruct

# 或使用 Git LFS
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ./qwen2.5-3b-instruct
```

### 3. 设置 CANN 环境

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

## 第二步：导出 ONNX 模型

### 基本用法

```bash
python3 export_qwen_to_onnx.py \
    --model ./qwen2.5-3b-instruct \
    --output qwen2.5-3b-instruct.onnx
```

### 完整参数说明

```bash
python3 export_qwen_to_onnx.py \
    --model Qwen/Qwen2.5-3B-Instruct   # 模型名称或本地路径
    --output qwen2.5-3b-instruct.onnx  # 输出ONNX文件路径
    --batch-size 1                      # 批大小（默认：1）
    --seq-len 1                         # 序列长度（默认：1）
    --max-past-seq-len 2048             # 最大历史序列长度（默认：2048）
    --device cpu                        # 目标设备：cpu/cuda/npu（默认：cpu）
```

### 模型配置说明

Qwen2.5-3B-Instruct 的关键配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 2048 | 隐藏层维度 |
| num_attention_heads | 16 | 注意力头数量 |
| num_key_value_heads | 4 | KV 头数量（GQA） |
| num_hidden_layers | 36 | Transformer 层数 |
| head_dim | 128 | 每个头的维度 |
| vocab_size | 151936 | 词表大小 |

### KV Cache 动态输入

ONNX 模型包含以下动态输入：

- **input_ids**: `(batch_size, seq_len)` - 输入 token IDs
- **attention_mask**: `(batch_size, 1, seq_len, total_seq_len)` - 注意力掩码
- **position_ids**: `(batch_size, seq_len)` - 位置 IDs
- **past_key_values_0 ~ past_key_values_35**: `(batch_size, 4, past_seq_len, 128)` - 36 层的 KV Cache

输出：
- **logits**: `(batch_size, seq_len, vocab_size)` - 预测 logits
- **present_key_values_0 ~ present_key_values_35**: 更新后的 KV Cache

## 第三步：转换为 OM 模型

### 基本用法

```bash
chmod +x convert_onnx_to_om.sh
./convert_onnx_to_om.sh
```

### 完整参数说明

```bash
./convert_onnx_to_om.sh \
    --onnx-model qwen2.5-3b-instruct.onnx  # 输入ONNX文件
    --output qwen2.5-3b-instruct            # 输出模型名称（不含.om）
    --soc-version Ascend310B1               # SoC版本（默认：Ascend310B1）
    --fp32                                   # 强制使用FP32（默认：FP16）
```

### ATC 命令详解

转换脚本使用的核心 ATC 命令参数：

```bash
atc \
    --model=qwen2.5-3b-instruct.onnx \
    --framework=5 \
    --output=qwen2.5-3b-instruct \
    --soc_version=Ascend310B1 \
    --input_format=NHWC \
    --precision_mode=allow_fp32_to_fp16 \
    --op_select_implmode=high_performance \
    --optypename_for_implmode="Gelu" \
    --enable_small_channel=1 \
    --enable_compress_weight=0 \
    --autoTuneMode=0 \
    --input_shape="input_ids:1,1;position_ids:1,1" \
    --dynamic_batch_size="input_ids:-1,-1;position_ids:-1,-1" \
    --input_nodes="input_ids,attention_mask,position_ids"
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| --framework | 5 | 5 = ONNX |
| --soc_version | Ascend310B1 | 目标 NPU 型号 |
| --input_format | NHWC | 输入数据格式 |
| --precision_mode | allow_fp32_to_fp16 | 允许 FP32 转 FP16 |
| --op_select_implmode | high_performance | 高性能模式 |
| --enable_small_channel | 1 | 启用小通道优化 |
| --autoTuneMode | 0 | 自动调优模式（关闭） |

## 第四步：ACL 推理

### 设置环境变量

```bash
# 设置 ACL Python 环境
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/7.0.0/python/site-packages:$PYTHONPATH
```

### 运行 ACL 推理

```bash
# 基本用法
python3 acl_inference_qwen.py \
    --model qwen2.5-0.5b-instruct.om \
    --prompt "Hello, how are you?"

# 完整参数
python3 acl_inference_qwen.py \
    --model qwen2.5-0.5b-instruct.om \
    --device 0 \
    --prompt "你的问题" \
    --max-tokens 100 \
    --temperature 0.7
```

### 预期输出

```
============================================================
Qwen2.5-0.5B-Instruct ACL Inference
============================================================

Model Information:
  Inputs:  27
  Outputs: 25

[OK] Inference completed in 0.150s
[Output] Logits shape: (1, 1, 151936)
[Output] Next token (argmax): 17
```

### 使用 msame 工具验证

```bash
# 如果已安装 msame
msame --model qwen2.5-3b-instruct.om \
      --input /path/to/input_data \
      --output /path/to/output \
      --outBin YES
```

## 注意事项

### 内存限制

Ascend 310B NPU 有 16GB 内存，Qwen2.5-3B 模型：
- FP32: ~12GB
- FP16: ~6GB
- INT8: ~3GB（推荐量化）

### 优化建议

1. **使用 FP16**: 设置 `--precision_mode=allow_fp32_to_fp16`
2. **KV Cache 量化**: 对 KV Cache 使用 INT8 量化
3. **模型剪枝**: 移除不必要的层或头
4. **批处理**: 仅在内存充足时使用批大小 > 1

### 常见问题

**Q: 转换失败，提示内存不足**
A: 使用 `--fp32` 参数强制 FP32，或考虑使用更小的模型变体

**Q: 动态维度不支持**
A: ATC 对某些动态形状支持有限，可能需要固定输入形状

**Q: 精度下降**
A: 检查 `--precision_mode` 设置，FP16 可能导致轻微精度损失

## 推理示例代码

```python
import numpy as np
from aclpy import acl

# 初始化 ACL
acl.init()
context, ret = acl.rt.create_context(device_id=0)

# 加载 OM 模型
model_id, ret = acl.mdl.load_from_file("qwen2.5-3b-instruct.om")

# 创建数据集
dataset = acl.mdl.create_dataset()

# 准备输入
input_data = np.array([[1, 2, 3]], dtype=np.int32)  # 示例输入
# ... 添加数据到 dataset

# 执行推理
ret = acl.mdl.execute(model_id, dataset, dataset)

# 获取输出
# ... 从 dataset 获取输出

# 清理
acl.mdl.unload(model_id)
acl.rt.destroy_context(context)
acl.finalize()
```

## 参考资源

- [CANN 开发指南](https://www.hiascend.com/document)
- [ATC 工具说明](https://www.hiascend.com/document?tag=development)
- [Qwen2.5 模型文档](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [ACL 推理开发指南](../examples/README_ACL_Example.md)
