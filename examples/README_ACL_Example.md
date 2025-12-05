# ACL (Ascend Computing Language) ResNet50 推理示例详解

## 概述

本示例展示了如何在 Orange Pi AI Pro (Ascend 310/310B NPU) 上使用 Python ACL API 进行 ResNet50 模型推理。

## 关键步骤详解

### 1. acl.init() - 初始化 ACL 运行时

```python
ret = acl.init(None, 1)
```

**作用**: 初始化 ACL (Ascend Computing Language) 运行时库，这是使用 ACL 的第一步。

**参数**:
- `None`: 使用默认配置。也可以传入字典来自定义配置，如 `{"acl.log_level": "INFO"}`
- `1`: ACL 线程模式，1表示单线程模式(简化错误处理)，0表示多线程模式

**返回值**:
- `ACL_SUCCESS` (0): 初始化成功
- 其他值: 错误码

**注意事项**:
- 必须在调用任何其他 ACL API 之前调用
- 只能调用一次，多次调用会返回错误
- 初始化失败时，无法使用任何 ACL 功能

### 2. acl.rt.set_device() - 设置设备

```python
ret = acl.rt.set_device(device_id)
```

**作用**: 指定使用哪个 Ascend NPU 设备。

**参数**:
- `device_id`: 设备ID，通常为 0 (第一个 NPU)

**设备上下文管理**:
```python
# 创建上下文和流
self.context, ret = acl.rt.create_context(self.device_id)
self.stream, ret = acl.rt.create_stream(self.context)
```

- **Context (上下文)**: 管理设备内存和执行资源
- **Stream (流)**: 异步执行队列，确保操作按顺序执行

### 3. acl.mdl.load_from_file() - 加载模型

```python
self.model_id, self.model_desc = acl.mdl.load_from_file(self.model_path)
```

**作用**: 从 .om 文件加载离线模型到 NPU 设备。

**参数**:
- `model_path`: .om 模型文件的绝对路径

**返回值**:
- `model_id`: 模型在设备中的标识符，用于后续执行推理
- `model_desc`: 模型描述符，包含输入输出张量的元数据

**.om 格式说明**:
- **OM**: Offline Model (离线模型)
- 由 ATC (Ascend Tensor Compiler) 工具将 PyTorch/TensorFlow 模型转换而来
- 包含算子融合、图优化、量化等优化
- 针对昇腾硬件进行高度优化

**模型转换示例** (使用 ATC):
```bash
# 安装 ATC 工具后，执行转换命令
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50 \
    --soc_version=Ascend310 \
    --input_format=NCHW \
    --input_shape="input:1,3,224,224"
```

### 4. 内存管理

**设备内存分配**:
```python
buffer, ret = acl.rt.malloc(size, self.device_id)
```

**主机到设备内存复制**:
```python
ret = acl.rt.memcpy(
    device_buffer,      # 目标设备缓冲区
    size,               # 复制大小
    host_data,          # 源主机数据
    len(host_data),     # 源数据大小
    acl.MEMCPY_HOST_TO_DEVICE
)
```

**设备到主机内存复制**:
```python
ret = acl.rt.memcpy(
    host_buffer,
    size,
    device_buffer,
    size,
    acl.MEMCPY_DEVICE_TO_HOST
)
```

### 5. acl.mdl.execute() - 执行推理

```python
ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
```

**作用**: 在 NPU 上执行模型推理。

**参数**:
- `model_id`: 模型标识符
- `input_dataset`: 输入数据集
- `output_dataset`: 输出数据集

**数据集创建**:
```python
# 创建输入数据集
input_dataset = acl.mdl.create_dataset()
for buffer in self.input_buffers:
    data_item = acl.create_data_buffer(buffer)
    acl.mdl.add_dataset_tensor(input_dataset, acl.MDL_INPUT, data_item)
```

### 6. 资源清理

**清理顺序很重要**:
```python
# 1. 释放内存缓冲区
acl.rt.free(buffer)

# 2. 销毁模型
acl.mdl.destroy_model(self.model_id)

# 3. 销毁模型描述符
acl.mdl.destroy_desc(self.model_desc)

# 4. 销毁流和上下文
acl.rt.destroy_stream(self.stream)
acl.rt.destroy_context(self.context)

# 5. 重置设备
acl.rt.reset_device(self.device_id)
```

## 完整工作流程

```
1. acl.init() 初始化 ACL
   ↓
2. acl.rt.set_device() 设置设备
   ↓
3. 创建 Context 和 Stream
   ↓
4. acl.mdl.load_from_file() 加载 .om 模型
   ↓
5. 获取模型输入输出信息
   ↓
6. acl.rt.malloc() 分配设备内存
   ↓
7. 预处理输入数据
   ↓
8. acl.rt.memcpy() 主机→设备
   ↓
9. acl.mdl.execute() 执行推理
   ↓
10. acl.rt.memcpy() 设备→主机
    ↓
11. 清理所有资源
```

## ImageNet 分类示例

模型输出通常是 1000 维向量 (ImageNet 1000 类)，需要后处理：

```python
def postprocess(output_data):
    """后处理: 计算类别概率和 Top-5 预测"""
    # 假设输出为 (1, 1000) float32
    probabilities = output_data[0].reshape(1000)

    # Softmax 计算概率
    exp_values = np.exp(probabilities - np.max(probabilities))
    probabilities = exp_values / np.sum(exp_values)

    # Top-5 预测
    top5_indices = np.argsort(probabilities)[-5:][::-1]

    # ImageNet 类别名称 (需要加载 Label 文件)
    labels = load_imagenet_labels()

    for i, idx in enumerate(top5_indices):
        print(f"Top-{i+1}: {labels[idx]} ({probabilities[idx]:.4f})")
```

## 性能优化建议

1. **异步执行**: 使用 `acl.mdl.execute_async()` + `acl.rt.synchronize_stream()`
2. **内存复用**: 避免重复分配内存，复用缓冲区
3. **批量推理**: 一次处理多张图像提高吞吐量
4. **模型量化**: 使用 INT8 量化减小模型大小

## 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| ACL_ERROR_INVALID_PARAM | 参数无效 | 检查参数是否正确 |
| ACL_ERROR_NOT_INITIALIZED | ACL 未初始化 | 确保调用 acl.init() |
| ACL_ERROR_OUT_MEMORY | 内存不足 | 释放不需要的资源 |
| ACL_ERROR_FILE_NOT_FOUND | 文件不存在 | 检查模型文件路径 |
| ACL_ERROR_INVALID_FILE | 文件格式错误 | 检查 .om 文件是否正确 |

## 环境准备

1. **安装 CANN**:
```bash
# 下载 CANN 社区版
wget https://www.hiascend.com/software/cann/community
# 安装 CANN Toolkit
```

2. **安装 Python ACL**:
```bash
# 安装 PyACL
pip3 install numpy opencv-python
# 安装 ACL 提供的 Python 绑定
```

3. **模型转换**:
```bash
# 使用 ATC 转换模型
atc --model=resnet50.onnx --framework=5 --output=resnet50 \
    --soc_version=Ascend310 --input_format=NCHW \
    --input_shape="input:1,3,224,224"
```

## 扩展阅读

- [CANN 开发文档](https://www.hiascend.com/document-center)
- [ACL Python API 参考](https://www.hiascend.com/document-center/zh/developer)
- [ATC 模型转换工具](https://www.hiascend.com/document-center/zh/developer)