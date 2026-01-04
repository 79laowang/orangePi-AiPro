# 快速开始指南

## 第一步：兼容性测试

```bash
cd ~/ai-works/orangePi-AiPro/vllm-ascend-chat
./test_docker.sh
```

这个脚本会：
- 检查系统兼容性
- 拉取 vLLM-Ascend Docker 镜像
- 运行完整的测试套件

预计耗时：10-20 分钟（首次需要下载镜像和模型）

## 第二步：启动 vLLM 服务器

### 方式 A：Docker (推荐)

```bash
./start_docker.sh
```

### 方式 B：原生安装

```bash
# 首先安装 vLLM-Ascend
pip install vllm-ascend

# 然后启动服务器
./start_server.sh
```

服务器启动后，你会看到：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 第三步：启动聊天应用

在**新的终端**中运行：

```bash
cd ~/ai-works/orangePi-AiPro/vllm-ascend-chat
python3 app_vllm.py
```

然后在浏览器打开：`http://localhost:7860`

## 预期性能

| 项目 | mindnlp | vLLM-Ascend |
|-----|---------|-------------|
| 推理速度 | 1.25 tokens/s | **10-30 tokens/s** |
| 提升倍数 | 1x | **8-24x** |

## 故障排除

### Docker 权限问题

```bash
sudo usermod -aG HwHiAiUser $USER
newgrp HwHiAiUser
```

### 模型下载慢

设置镜像加速：
```bash
export VLLM_USE_MODELSCOPE=true
```

### 内存不足

减小模型上下文：
```bash
export MAX_MODEL_LEN=1024
./start_server.sh
```
