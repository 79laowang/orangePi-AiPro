# MindSpore 安装指南 - Orange Pi AI Pro (Ascend 310B)

适用于 **CANN 7.1.0** + **Python 3.9** + **aarch64** Ubuntu 环境

---

## 目录

1. [系统限制说明](#系统限制说明)
2. [使用场景](#使用场景)
3. [环境检查](#环境检查)
4. [CANN 7.1.0 兼容性修复](#cann-710-兼容性修复)
5. [MindSpore 安装](#mindspore-安装)
6. [验证安装](#验证安装)
7. [常见问题](#常见问题)

---

## 系统限制说明

**Orange Pi AI Pro (Ascend 310B) 使用 NPU 进行大模型推理时存在内存限制：**

| 配置项 | 值 |
|--------|-----|
| 系统内存 | 15GB RAM |
| NPU 共享内存需求 | 8-10GB (不可 swap) |
| CANN Worker 进程 | 8-10 个 × ~1GB/进程 |
| **结论** | **NPU 模式不适合大模型推理** |

**推荐方案**:
- ✅ 使用 CPU 模式进行推理 (稳定可靠)
- ✅ 使用 transformers + PyTorch 直接加载模型
- ❌ 避免使用 NPU 模式进行大模型推理 (会 OOM)

---

## 使用场景

### ✅ 可以做的场景

#### 1. CPU 模式大模型推理

**适用模型**:
- Qwen2-1.5B / Qwen2-0.5B (中文对话/创作)
- Llama-3.2-1B / 3B (英文对话)
- 其他 1-3B 参数量的语言模型

**使用方式**:
```bash
# 方式一: MindSpore CPU 模式
import mindspore
mindspore.set_context(device_target="CPU", mode=mindspore.PYNATIVE_MODE)

# 方式二: transformers + PyTorch (推荐)
pip install transformers torch sentencepiece
python3 infer_qwen_cpu.py
```

**典型应用**:
- 中文小说创作 (武侠、仙侠、都市)
- 智能对话助手
- 文本摘要/翻译
- 代码生成/补全

---

#### 2. NPU 模式小模型推理

**适用模型** (单个模型文件 < 1GB):
- ResNet-50/101 (图像分类)
- YOLOv5/v8 (目标检测)
- MobileNet (轻量级图像分类)
- BERT-Base (NLP 分类任务)

**使用方式**:
```python
import mindspore
mindspore.set_context(device_target="Ascend")

# 加载小模型进行推理
from mindspore import Tensor
import numpy as np

# 示例: 图像分类
input_tensor = Tensor(np.random.rand(1, 3, 224, 224).astype(np.float32))
output = model(input_tensor)
```

**典型应用**:
- 人脸识别
- 车牌识别
- 工业质检
- 智能监控

---

#### 3. 边缘计算场景

**特点**: 低功耗、实时响应、离线运行

**应用场景**:
- 智能家居控制
- 机器人视觉导航
- 无人机图像处理
- 智能安防系统

**优势**:
- 功耗 < 20W
- 无需联网
- 数据隐私保护

---

### ❌ 不能做的场景

#### 1. NPU 模式大模型推理

**原因**: 内存限制 (15GB RAM < 8-10GB NPU 共享内存需求)

**不支持的模型**:
- Qwen2-7B 及以上
- Llama-3-8B 及以上
- 任何需要 > 3GB 内存的大模型

**错误表现**:
```
Killed (Exit Code 137)
dmesg: Memory cgroup out of memory: shmem-rss: 10354468kB
```

**替代方案**: 使用 CPU 模式

---

#### 2. NPU 模式模型训练

**原因**:
- Ascend 310B 是推理专用芯片
- 缺少训练所需的高精度计算单元
- 内存不足以存储梯度/优化器状态

**不支持的操作**:
- 微调 (Fine-tuning)
- LoRA 训练
- 全量训练

**替代方案**:
- 在云端/高性能服务器训练
- 下载预训练模型直接推理

---

#### 3. 大批量并行推理

**原因**: 内存限制，无法同时加载多个模型

**限制**:
- 无法同时运行多个 NPU 推理进程
- 批处理大小 (batch size) 受限

**替代方案**:
- 串行推理
- 使用 CPU 模式 (可并发多个进程)

---

### 📊 场景选择指南

| 场景 | 推荐方案 | 模式 | 预期性能 |
|------|----------|------|----------|
| 中文小说创作 | transformers + CPU | CPU | 10-20 tokens/s |
| 图像分类 | MindSpore + NPU | NPU | 50-100 fps |
| 目标检测 | MindSpore + NPU | NPU | 20-30 fps |
| 智能对话 | transformers + CPU | CPU | 15-25 tokens/s |
| 人脸识别 | MindSpore + NPU | NPU | 30-50 fps |
| 代码生成 | transformers + CPU | CPU | 8-15 tokens/s |

---

## 环境检查

```bash
# 检查系统架构
uname -m
# 期望输出: aarch64

# 检查 Python 版本
python3 --version
# 期望输出: Python 3.9.x

# 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/runtime/version.info | grep "^Version="
# 期望输出: Version=7.1.0.x.x
```

---

## CANN 7.1.0 兼容性修复

CANN 7.1.0 与 MindSpore 2.x 存在已知兼容性问题，**必须在安装 MindSpore 之前修复**。

### 问题说明

**错误信息**:
```
AttributeError: module 'ascend_toolkit.tbe.common.utils.op_tiling' has no attribute 'sys_version'
```
或
```
NameError: name 'sys_version' is not defined
```

**根本原因**:
- 文件: `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py`
- 行号: ~1008
- 问题: `sys_version` 变量被使用但未定义，导致模块导入失败

### 修复步骤

```bash
# 进入项目目录
cd /home/HwHiAiUser/ai-works/orangePi-AiPro

# 运行补丁脚本
python3 patch_op_tiling.py

# 应用修复（需要 sudo 权限）
sudo cp ./op_tiling_patched.py /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py

# 验证修复
grep "sys_version = \"linux\"" /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py
# 期望输出: sys_version = "linux"  # Default OS version for Ascend platform
```

### 恢复原始文件（如需要）

```bash
sudo cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py.cann_fix_backup \
      /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py
```

### 手动修复（如补丁工具不可用）

如果 `patch_op_tiling.py` 不可用，可以手动编辑文件：

```bash
# 1. 备份原始文件
sudo cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py \
        /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py.manual_fix_backup

# 2. 编辑文件（在 ~1008 行附近添加）
sudo nano /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py
```

在文件中找到类似以下内容的位置（约 1008 行）：
```python
# 原始代码
def some_function():
    ...
    if sys_version == "linux":  # <-- sys_version 未定义！
        ...
```

在文件开头添加变量定义：
```python
# 在文件顶部的导入语句后添加
import sys
sys_version = "linux"  # Default OS version for Ascend platform
```

或者在报错行前添加条件判断：
```python
if 'sys_version' not in locals():
    sys_version = "linux"
```

保存后验证修复：
```bash
python3 -c "from tbe.common.utils import op_tiling; print('✓ 修复成功')"
```

---

## MindSpore 安装

### 方法一：使用自动安装脚本

```bash
cd /home/HwHiAiUser/ai-works/orangePi-AiPro
chmod +x setup_mindspore.sh
echo "1" | bash setup_mindspore.sh
```

### 方法二：手动安装

```bash
# 设置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 下载 MindSpore 2.2.14（支持 CANN 7.x）
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/aarch64/mindspore-2.2.14-cp39-cp39-linux_aarch64.whl

# 安装 MindSpore
pip3 install mindspore-2.2.14-cp39-cp39-linux_aarch64.whl --user

# 安装 CANN Python 依赖
pip3 install --user /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*.whl
pip3 install --user /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*.whl
```

---

## 验证安装

### 设置环境变量

```bash
# 创建环境配置文件
cat > ~/.mindspore_env << 'EOF'
# MindSpore + CANN 环境变量
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=${ASCEND_HOME}/opp
export LD_LIBRARY_PATH=${ASCEND_HOME}/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=${ASCEND_HOME}/python/site-packages:${PYTHONPATH}
EOF

# 添加到 ~/.bashrc
echo "source ~/.mindspore_env" >> ~/.bashrc

# 立即生效
source ~/.mindspore_env
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

### 步骤 1: 验证 CANN 修复

```bash
# 检查补丁是否应用成功
grep "sys_version = \"linux\"" /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py
# 期望输出: sys_version = "linux"  # Default OS version for Ascend platform

# 测试 CANN 模块导入
python3 -c "from tbe.common.utils import op_tiling; print('✓ CANN 模块导入成功')"
# 期望输出: ✓ CANN 模块导入成功
```

**如果失败**: 重新运行补丁脚本（见上方 CANN 7.1.0 兼容性修复章节）

---

### 步骤 2: 验证 MindSpore 安装

```bash
# 检查 MindSpore 版本
python3 -c "import mindspore; print('MindSpore 版本:', mindspore.__version__)"
# 期望输出: MindSpore 版本: 2.2.14

# 运行 MindSpore 内置测试
python3 -c "import mindspore; mindspore.run_check()"
# 期望输出: MindSpore version: 2.2.14
#           The result of multiplication calculation is correct, MindSpore works well!
```

---

### 步骤 3: 验证 CPU 模式 (推荐)

```bash
python3 << 'EOF'
import mindspore
import numpy as np

# 设置 CPU 模式
mindspore.set_context(device_target="CPU", mode=mindspore.PYNATIVE_MODE)
print("✓ 设备模式: CPU (PYNATIVE)")

# 简单计算测试
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
result = ops.add(x, y)

print("✓ CPU 推理测试: 成功")
print(f"✓ 计算结果: {result[0, 0, 0, 0]}")
EOF
# 期望输出: 计算结果: 2.0
```

---

### 步骤 4: 验证 NPU 模式 (可能失败)

**警告**: 由于系统内存限制 (15GB RAM < 8-10GB NPU 共享内存需求)，此测试**可能会失败**。

```bash
# 检查 NPU 设备信息
npu-smi info
# 期望输出: NPU 设备信息表

# 尝试 NPU 推理测试
python3 << 'EOF'
import mindspore
import numpy as np

# 使用 Ascend 设备
mindspore.set_context(device_target="Ascend")

# 简单计算测试
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
result = ops.add(x, y)
print("✓ NPU 推理测试: 成功")
print(f"✓ 计算结果: {result[0, 0, 0, 0]}")
EOF
```

**可能的结果**:
- ✅ **成功**: 输出 `NPU 推理测试: 成功` 和 `计算结果: 2.0`
- ❌ **失败**: 进程被 kill，退出码 137

**如果失败** (Exit Code 137):
```bash
# 查看 OOM 日志
sudo dmesg | tail -20 | grep -i "killed\|oom"

# 解决方案: 使用 CPU 模式 (见步骤 3)
```

---

### 步骤 5: 系统资源检查

```bash
# 检查内存信息
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree"
# 期望输出:
#   MemTotal:       15984680 kB  (~15GB)
#   MemAvailable:   13000000 kB  (~12-13GB 可用)
#   SwapTotal:      10485752 kB  (~10GB)

# 检查共享内存限制
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
# 期望输出: 9223372036854771712 (无限制)

# 检查 NPU 设备
npu-smi info
```

---

### 验证总结

| 测试项 | 状态 | 说明 |
|--------|------|------|
| CANN 补丁 | ✅ 必须通过 | 否则无法导入模块 |
| MindSpore 版本 | ✅ 必须通过 | 确认安装成功 |
| CPU 模式 | ✅ 推荐使用 | 稳定可靠 |
| NPU 模式 | ⚠️ 可能 OOM | 受内存限制 |

**推荐配置**: 使用 CPU 模式进行开发

---

## 常见问题

### Q1: `cannot import name 'utils' from partially initialized module 'tbe.common'`

**原因**: CANN 7.1.0 兼容性问题未修复

**解决**:
```bash
# 重新运行补丁脚本
cd /home/HwHiAiUser/ai-works/orangePi-AiPro
python3 patch_op_tiling.py
sudo cp ./op_tiling_patched.py /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py
```

### Q2: `NameError: name 'sys_version' is not defined`

**原因**: 同上，补丁未应用

**解决**: 同 Q1

### Q3: NPU 推理报错但 CPU 模式正常

**原因**: CANN 环境变量未正确设置

**解决**:
```bash
source ~/.mindspore_env
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Q4: 想使用 CPU 模式而非 NPU

**解决**: 在代码中设置：
```python
import mindspore
mindspore.set_context(device_target="CPU")
```

### Q5: NPU 模式 OOM (Exit Code 137)

**症状**: 运行 NPU 推理时进程被系统 kill，退出码 137

**根本原因**:
- MindSpore NPU 模式通过 CANN 框架启动多个 ForkServerPoolWorker 进程
- 每个进程消耗 ~1GB 共享内存 (shmem-rss)
- 系统共 15GB RAM，NPU 需要 8-10GB 共享内存
- **共享内存无法使用 swap**，必须全部在物理 RAM 中
- Ascend 310B 设计用于推理，非大模型训练

**dmesg 日志示例**:
```
Memory cgroup out of memory: Killed process 15976 (python3)
shmem-rss: 10354468kB (~10GB per process)
```

**解决方案**:

**推荐方案: 使用 CPU 模式**
```python
import mindspore
mindspore.set_context(device_target="CPU", mode=mindspore.PYNATIVE_MODE)
```

或者使用 transformers + CPU (更稳定):
```bash
pip install transformers torch sentencepiece
python3 infer_qwen_cpu.py
```

**替代方案** (不推荐):
1. 使用更小模型 (Qwen2-0.5B，内存占用 ~3GB)
2. 减少并发 worker 数 (效果有限)
3. 升级硬件 (32GB+ RAM)

### Q6: 如何切换 MindSpore 版本

```bash
# 卸载当前版本
pip3 uninstall mindspore -y

# 安装新版本
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/unified/aarch64/mindspore-{version}-cp39-cp39-linux_aarch64.whl
pip3 install mindspore-{version}-cp39-cp39-linux_aarch64.whl --user
```

### Q7: 中文小说生成推荐方案

**CPU 推理脚本** (已提供):
- `infer_qwen_cpu.py`: 完整的交互式小说创作工具
- 支持 武侠/仙侠/都市 多种风格
- 使用 Qwen2-1.5B-Instruct 模型

**运行方式**:
```bash
# 1. 安装依赖
pip install transformers torch sentencepiece

# 2. 运行脚本
python3 infer_qwen_cpu.py

# 选择模式:
#   1. 演示模式 (快速体验)
#   2. 交互模式 (自由创作)
```

---

## 版本兼容性表

| MindSpore | CANN | 状态 |
|-----------|------|------|
| 2.2.14 | 7.0 / 7.1 | ✅ 推荐 |
| 2.3.x | 7.x | ⚠️ 需验证 |
| 2.5.x | 8.0 | ⚠️ 需升级 CANN |
| 2.6.x | 8.1 | ⚠️ 需升级 CANN |
| 2.7.x | 8.2RC1 | ⚠️ 需升级 CANN |

**当前环境**: CANN 7.1.0.3.220 → 使用 MindSpore 2.2.14

---

## 下一步

安装完成后，可以：

1. **下载 Qwen2 模型**: `python3 download_qwen_model.py`
2. **运行推理测试**: `python3 infer_qwen_lite.py`
3. **查看 MindSpore 文档**: https://www.mindspore.cn/docs

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `setup_mindspore.sh` | 自动安装脚本 |
| `patch_op_tiling.py` | CANN 兼容性补丁工具 |
| `op_tiling_patched.py` | 修复后的 op_tiling.py |
| `op_tiling.py.bak` | 原始文件备份 |
| `fix_cann_env.sh` | 环境变量修复脚本 |
| `infer_qwen_cpu.py` | CPU 模式小说推理脚本 |
| `novel_inference_cpu.py` | CPU 推理方案说明文档 |
| `download_qwen_model.py` | Qwen2 模型下载工具 |
| `convert_qwen_to_mindspore.py` | Qwen2 转 MindSpore 格式工具 |

---

## 总结

**重要结论**:
1. NPU 模式不适合大模型 (如 Qwen2-1.5B) 推理 - 内存限制
2. CPU 模式稳定可靠 - 推荐使用
3. transformers + PyTorch 直接加载模型是最简单方案

**快速开始 (CPU 推理)**:
```bash
pip install transformers torch sentencepiece
python3 infer_qwen_cpu.py
```
