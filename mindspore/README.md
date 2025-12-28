# MindSpore on Orange Pi AI Pro

本目录包含在 Orange Pi AI Pro (Ascend 310B) 上使用 MindSpore 的相关文档、脚本和工具。

## 目录结构

```
mindspore/
├── docs/           # 文档
│   └── MINDSPORE_INSTALL_GUIDE.md    # 完整安装和使用指南
├── scripts/        # 安装和补丁脚本
│   ├── setup_mindspore.sh            # 自动安装脚本
│   ├── patch_op_tiling.py            # CANN 7.1.0 兼容性补丁工具
│   ├── op_tiling_patched.py          # 修复后的 op_tiling.py
│   └── fix_cann_env.sh               # 环境变量修复脚本
└── tools/          # 工具
    └── download_qwen_model.py        # Qwen2 模型下载工具
```

## 快速开始

### 1. 阅读安装指南

```bash
cat docs/MINDSPORE_INSTALL_GUIDE.md
```

### 2. 安装 MindSpore

```bash
cd scripts
chmod +x setup_mindspore.sh
./setup_mindspore.sh
```

### 3. 运行 CPU 推理（推荐）

参考项目根目录下的推理脚本：
```bash
cd /home/HwHiAiUser/ai-works/orangePi-AiPro
pip install transformers torch sentencepiece
python3 infer_qwen_cpu.py
```

## 系统限制

| 配置项 | 值 |
|--------|-----|
| 系统内存 | 15GB RAM |
| NPU 共享内存需求 | 8-10GB (不可 swap) |
| **结论** | **NPU 模式不适合大模型推理** |

## 已知问题

### CANN 7.1.0 兼容性问题

**症状**: `AttributeError: module 'ascend_toolkit.tbe.common.utils.op_tiling' has no attribute 'sys_version'`

**解决**: 使用提供的补丁工具
```bash
cd scripts
python3 patch_op_tiling.py
sudo cp ./op_tiling_patched.py /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py
```

### NPU OOM (Exit Code 137)

**症状**: 运行 NPU 推理时进程被系统 kill

**解决**: 使用 CPU 模式替代
```python
import mindspore
mindspore.set_context(device_target="CPU", mode=mindspore.PYNATIVE_MODE)
```

## 推荐使用场景

| 场景 | 推荐方案 | 预期性能 |
|------|----------|----------|
| 中文小说创作 | transformers + CPU | 10-20 tokens/s |
| 图像分类 | MindSpore + NPU | 50-100 fps |
| 智能对话 | transformers + CPU | 15-25 tokens/s |
| 人脸识别 | MindSpore + NPU | 30-50 fps |

## 相关链接

- [MindSpore 官方文档](https://www.mindspore.cn/docs)
- [CANN 文档](https://www.hiascend.com/document)
- [Qwen 模型](https://github.com/QwenLM/Qwen)
