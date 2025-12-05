# ACL 快速参考卡

## 核心 API 速查

### 初始化和配置

| 操作 | API | 说明 |
|------|-----|------|
| 初始化 ACL | `acl.init()` | 必须第一步，只调用一次 |
| 设置设备 | `acl.rt.set_device(id)` | 指定使用哪个 NPU (id=0,1,2...) |
| 创建上下文 | `acl.rt.create_context(id)` | 管理设备资源 |
| 创建流 | `acl.rt.create_stream(ctx)` | 异步执行队列 |

### 模型加载

| 操作 | API | 说明 |
|------|-----|------|
| 加载模型 | `acl.mdl.load_from_file(path)` | 加载 .om 文件，返回 (model_id, model_desc) |
| 获取输入数 | `acl.mdl.get_num_inputs(desc)` | 获取模型输入张量数量 |
| 获取输出数 | `acl.mdl.get_num_outputs(desc)` | 获取模型输出张量数量 |
| 获取输入尺寸 | `acl.mdl.get_input_size_by_index(desc, i)` | 获取第 i 个输入的大小 |
| 获取输出尺寸 | `acl.mdl.get_output_size_by_index(desc, i)` | 获取第 i 个输出的大小 |

### 内存管理

| 操作 | API | 说明 |
|------|-----|------|
| 分配内存 | `acl.rt.malloc(size, device_id)` | 在 NPU 设备上分配内存 |
| 复制 H→D | `acl.rt.memcpy(dst, dst_size, src, src_size, MEMCPY_HOST_TO_DEVICE)` | 主机到设备 |
| 复制 D→H | `acl.rt.memcpy(dst, dst_size, src, src_size, MEMCPY_DEVICE_TO_HOST)` | 设备到主机 |
| 释放内存 | `acl.rt.free(buffer)` | 释放设备内存 |

### 推理执行

| 操作 | API | 说明 |
|------|-----|------|
| 创建数据集 | `acl.mdl.create_dataset()` | 创建数据集对象 |
| 添加数据项 | `acl.mdl.add_dataset_tensor(dataset, type, buffer)` | type=MDL_INPUT 或 MDL_OUTPUT |
| 执行推理 | `acl.mdl.execute(model_id, input_dataset, output_dataset)` | 同步执行 |
| 销毁数据集 | `acl.mdl.destroy_dataset(dataset)` | 清理数据集 |

### 清理资源

| 操作 | API | 说明 |
|------|-----|------|
| 销毁模型 | `acl.mdl.destroy_model(model_id)` | 销毁模型 |
| 销毁描述符 | `acl.mdl.destroy_desc(desc)` | 销毁模型描述符 |
| 销毁流 | `acl.rt.destroy_stream(stream)` | 销毁流 |
| 销毁上下文 | `acl.rt.destroy_context(context)` | 销毁上下文 |
| 重置设备 | `acl.rt.reset_device(device_id)` | 重置设备 |

## 最小工作流程 (10 步)

```python
# 1. 初始化
acl.init()

# 2. 设置设备
acl.rt.set_device(0)

# 3. 创建环境
context = acl.rt.create_context(0)
stream = acl.rt.create_stream(context)

# 4. 加载模型
model_id, model_desc = acl.mdl.load_from_file("model.om")

# 5. 获取信息
input_num = acl.mdl.get_num_inputs(model_desc)
output_num = acl.mdl.get_num_outputs(model_desc)

# 6. 分配内存
input_buffers = [acl.rt.malloc(size, 0) for size in input_sizes]
output_buffers = [acl.rt.malloc(size, 0) for size in output_sizes]

# 7. 复制输入
acl.rt.memcpy(input_buffers[0], size, host_data, len(host_data), MEMCPY_HOST_TO_DEVICE)

# 8. 创建数据集
input_dataset = acl.mdl.create_dataset()
output_dataset = acl.mdl.create_dataset()
# ... 添加数据项

# 9. 执行
acl.mdl.execute(model_id, input_dataset, output_dataset)

# 10. 复制输出
acl.rt.memcpy(host_output, size, output_buffers[0], size, MEMCPY_DEVICE_TO_HOST)

# 清理
acl.mdl.destroy_dataset(input_dataset)
acl.mdl.destroy_dataset(output_dataset)
acl.rt.free(input_buffers)
acl.rt.free(output_buffers)
acl.mdl.destroy_model(model_id)
acl.mdl.destroy_desc(model_desc)
acl.rt.destroy_stream(stream)
acl.rt.destroy_context(context)
acl.rt.reset_device(0)
```

## 数据类型转换

### 输入数据准备
```python
# 假设输入形状 (1, 3, 224, 224)
image = np.random.random((1, 3, 224, 224)).astype(np.float32)

# 复制到设备
input_bytes = image.tobytes()
acl.rt.memcpy(input_buffer, input_size, input_bytes, len(input_bytes), MEMCPY_HOST_TO_DEVICE)
```

### 输出数据读取
```python
# 从设备复制到主机
output_bytes = np.zeros(output_size, dtype=np.uint8)
acl.rt.memcpy(output_bytes, output_size, output_buffer, output_size, MEMCPY_DEVICE_TO_HOST)

# 转换为原始形状
# 例如: 输出可能是 (1, 1000)
output = output_bytes.view(np.float32).reshape(1, 1000)
```

## 错误处理

```python
def check_ret(ret, func_name):
    if ret != acl.ACL_SUCCESS:
        raise RuntimeError(f"{func_name} 失败: {ret}")

# 使用
ret = acl.rt.set_device(0)
check_ret(ret, "acl.rt.set_device")
```

## 常用常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `acl.ACL_SUCCESS` | 0 | 成功 |
| `acl.MEMCPY_HOST_TO_DEVICE` | 1 | 主机到设备 |
| `acl.MEMCPY_DEVICE_TO_HOST` | 2 | 设备到主机 |
| `acl.MDL_INPUT` | 0 | 输入张量 |
| `acl.MDL_OUTPUT` | 1 | 输出张量 |

## 模型转换命令

### ATC 转换 ONNX
```bash
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50 \
    --soc_version=Ascend310 \
    --input_format=NCHW \
    --input_shape="input:1,3,224,224"
```

### ATC 转换 PyTorch (.pth)
```bash
# 先转换为 ONNX
torch.onnx.export(model, dummy_input, "resnet50.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

# 再转换为 OM
atc --model=resnet50.onnx --framework=5 --output=resnet50 \
    --soc_version=Ascend310 --input_format=NCHW \
    --input_shape="input:-1,3,224,224"
```

## 性能优化技巧

1. **异步执行**
   ```python
   acl.mdl.execute_async(model_id, input_dataset, output_dataset, stream)
   acl.rt.synchronize_stream(stream)
   ```

2. **内存复用**
   ```python
   # 推理前分配一次，推理后不释放，复用缓冲区
   # 适用于批量推理场景
   ```

3. **批量推理**
   ```python
   # 输入形状从 (1, 3, 224, 224) 改为 (4, 3, 224, 224)
   # 一次处理 4 张图像，吞吐量提升 4 倍
   ```

4. **动态 Batch**
   ```bash
   # 转换时使用 -1 表示动态维度
   --input_shape="input:-1,3,224,224"
   ```

## 调试技巧

1. **检查模型信息**
   ```python
   for i in range(input_num):
       dims = acl.mdl.get_input_dims(model_desc, i)
       dtype = acl.mdl.get_input_data_type(model_desc, i)
       print(f"输入 {i}: {dims}, {dtype}")
   ```

2. **检查内存**
   ```python
   # 查看设备内存使用
   import subprocess
   result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
   print(result.stdout)
   ```

3. **日志级别**
   ```python
   # 设置 ACL 日志级别
   config = {"acl.log_level": "INFO"}
   acl.init(config)
   ```

## 资源

- [CANN 社区版下载](https://www.hiascend.com/software/cann/community)
- [ACL Python API 文档](https://www.hiascend.com/document-center/zh/developer)
- [ATC 模型转换工具](https://www.hiascend.com/document-center/zh/developer)
- [昇腾开发者社区](https://www.hiascend.com/developer)