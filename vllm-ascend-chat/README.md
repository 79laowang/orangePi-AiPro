# Qwen Chatbot with vLLM-Ascend
## High-Performance LLM Inference on Orange Pi AI Pro

基于 vLLM-Ascend 的高性能 Qwen 聊天机器人，支持 Flash Attention 和 KV Cache 优化。

## 🚀 性能对比

| 推理引擎 | tokens/s | 相比提升 |
|---------|----------|---------|
| mindnlp (baseline) | 1.25 | 1x |
| vLLM-Ascend (预期) | 10-30 | **8-24x** |

## 📋 系统要求

### 硬件
- **开发板**: Orange Pi AI Pro
- **NPU**: Ascend 310B4
- **内存**: 至少 4GB 可用 RAM

### 软件
- **操作系统**: Ubuntu 22.04.3 LTS (Kernel 5.10.0+)
- **Docker**: 20.10+ (推荐) 或 Python 3.10+
- **CANN**: 8.1.RC1 或更高版本

## 🔧 安装方法

### 方法 A: Docker (推荐)

Docker 容器包含所有依赖，开箱即用。

```bash
# 1. 运行测试脚本
./test_docker.sh

# 2. 启动 vLLM 服务器
docker run --rm \
  --name vllm-ascend-server \
  --shm-size=2g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /root/.cache:/root/.cache \
  -p 8000:8000 \
  -e ASCEND_VISIBLE_DEVICES=0 \
  quay.io/ascend/vllm-ascend:v0.11.0rc1 \
  vllm serve Qwen/Qwen2.5-0.5B-Instruct

# 3. 在另一个终端启动聊天应用
python3 app_vllm.py
```

### 方法 B: pip 安装

```bash
# 1. 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 安装 vLLM-Ascend
pip install vllm-ascend

# 3. 运行测试
python3 test_vllm.py

# 4. 启动 vLLM 服务器
vllm serve Qwen/Qwen2.5-0.5B-Instruct &

# 5. 启动聊天应用
python3 app_vllm.py
```

## 📂 文件结构

```
vllm-ascend-chat/
├── README.md              # 本文件
├── test_docker.sh         # Docker 兼容性测试脚本
├── test_vllm.py           # Python 测试脚本
├── app_vllm.py            # Gradio 聊天应用
├── start_server.sh        # 启动 vLLM 服务器脚本
└── docker-compose.yml     # Docker Compose 配置
```

## 🎯 快速开始

### 1. 兼容性测试

首先确认你的系统支持 vLLM-Ascend：

```bash
chmod +x test_docker.sh
./test_docker.sh
```

测试包括：
- Docker 和 NPU 设备检查
- vLLM-Ascend 镜像拉取
- 模型初始化测试
- 单次和批量生成测试

### 2. 启动服务器

```bash
# 使用启动脚本 (推荐)
chmod +x start_server.sh
./start_server.sh

# 或手动启动
vllm serve Qwen/Qwen2.5-0.5B-Instruct
```

服务器将在 `http://0.0.0.0:8000` 启动。

### 3. 启动聊天应用

```bash
python3 app_vllm.py
```

然后在浏览器打开: `http://localhost:7860`

## 🔍 vLLM-Ascend 优势

| 特性 | mindnlp | vLLM-Ascend |
|-----|---------|-------------|
| Flash Attention | ❌ | ✅ |
| PagedAttention (KV Cache) | ❌ | ✅ |
| 连续批处理 | ❌ | ✅ |
| OpenAI API 兼容 | ❌ | ✅ |
| 推理速度 | 1.25 tokens/s | **10-30 tokens/s** |

## 📊 性能测试

运行完整性能测试：

```bash
# Docker 方式
./test_docker.sh

# Python 方式
python3 test_vllm.py
```

预期输出：
```
[Test 1/4] Importing vLLM...
✅ vLLM imported successfully

[Test 2/4] Initializing model...
✅ Model initialized successfully

[Test 3/4] Testing simple generation...
✅ Generation successful!
   Speed: 15.23 tokens/s

[Test 4/4] Testing Chinese generation...
✅ Chinese generation successful!
   Speed: 14.87 tokens/s

🎉 All tests passed!
```

## ⚠️ 已知问题

### Ascend 310B4 兼容性

**状态**: 实验性支持

vLLM-Ascend 官方支持硬件列表：
- ✅ Atlas A2 系列
- ✅ Atlas 800I A2
- ✅ Atlas A3 系列
- ⚠️ **Ascend 310B4** (Orange Pi AI Pro)

**注意事项**:
- 功能应该可以正常工作
- 性能可能不如官方支持硬件
- 某些高级功能可能不支持

### 内存限制

Ascend 310B4 NPU 内存约 15GB，建议：
- 使用较小的模型 (≤1B 参数)
- 设置 `max_model_len=2048` 或更小
- 避免同时运行多个大型模型

## 🛠️ 故障排除

### 问题: Docker 容器无法访问 NPU

**症状**:
```
PermissionError: [Errno 13] Permission denied: '/dev/davinci0'
```

**解决方案**:
```bash
# 将用户添加到 HwHiAiUser 组
sudo usermod -aG HwHiAiUser $USER

# 重新登录或运行
newgrp HwHiAiUser
```

### 问题: 模型下载失败

**症状**:
```
OSError: Can't load tokenizer for 'Qwen/Qwen2.5-0.5B-Instruct'
```

**解决方案**:
```bash
# 使用镜像加速
export VLLM_USE_MODELSCOPE=true

# 或设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题: Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```bash
# 减小模型上下文长度
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 1024
```

## 📖 相关资源

- [vLLM-Ascend 官方文档](https://docs.vllm.ai/projects/ascend/zh-cn/latest/)
- [vLLM-Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [Qwen 模型](https://huggingface.co/Qwen)
- [昇腾社区](https://www.hiascend.com/)

## 📄 许可证

本项目遵循 Apache 2.0 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**最后更新**: 2026-01-04
