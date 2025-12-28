#!/bin/bash
# MindSpore 安装脚本 - Orange Pi AI Pro (Ascend 310B, CANN 7.1.0)
# Python 3.9, aarch64, Ubuntu 22.04

set -e

echo "======================================"
echo "MindSpore 安装向导"
echo "======================================"
echo "检测环境信息..."

# 检测 Python 版本
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 检测架构
ARCH=$(uname -m)
echo "系统架构: $ARCH"

# 检测 CANN 版本
CANN_VERSION=$(cat /usr/local/Ascend/ascend-toolkit/7.0.0/runtime/version.info | grep "^Version=" | cut -d'=' -f2)
echo "CANN 版本: $CANN_VERSION"

echo ""
echo "======================================"
echo "推荐安装方案"
echo "======================================"
echo ""
echo "检测到 CANN 7.1.0"
echo ""
echo "推荐安装方案:"
echo ""
echo "方案 1: MindSpore 2.2.14 (推荐，支持 CANN 7.x)"
echo "  - 兼容当前 CANN 版本"
echo "  - 支持 Ascend 310B 推理"
echo ""
echo "方案 2: 跳过安装，手动处理"
echo ""

read -p "选择方案 (1/2) [1]: " choice
choice=${choice:-1}

# 设置环境变量（CANN）
export PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

echo ""
echo "设置 CANN 环境变量..."

# 创建 MindSpore 版本选择
if [ "$choice" = "1" ]; then
    # MindSpore 2.2.14 - 支持 CANN 7.x
    MS_VERSION="2.2.14"
    BASE_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com"
    WHEEL_PATH="/MindSpore/unified/aarch64"
else
    echo "跳过安装"
    exit 0
fi

echo ""
echo "======================================"
echo "安装 MindSpore $MS_VERSION"
echo "======================================"
echo ""

# MindSpore wheel 包名格式: mindspore-{version}-cp{python}-cp{python}-linux_{arch}.whl
# Python 3.9 -> cp39
# aarch64 -> aarch64
# OBS URL 格式: https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/unified/aarch64/

WHEEL_NAME="mindspore-${MS_VERSION}-cp39-cp39-linux_aarch64.whl"
DOWNLOAD_URL="${BASE_URL}/${MS_VERSION}${WHEEL_PATH}/${WHEEL_NAME}"

echo "下载地址:"
echo "  $DOWNLOAD_URL"
echo ""

# 下载并安装
DOWNLOAD_DIR="./mindspore_install"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "正在下载..."
wget -q "$DOWNLOAD_URL" -O "$WHEEL_NAME" || {
    echo ""
    echo "======================================"
    echo "下载失败！请尝试手动安装："
    echo "======================================"
    echo ""
    echo "1. 访问 MindSpore 官网下载页面:"
    echo "   https://www.mindspore.cn/install"
    echo ""
    echo "2. 选择以下配置:"
    echo "   - 硬件平台: Ascend"
    echo "   - 软件环境: Linux-aarch64-Python3.9"
    echo "   - CANN 版本: 7.0/7.1"
    echo "   - MindSpore 版本: $MS_VERSION"
    echo ""
    echo "3. 下载 wheel 文件后，运行:"
    echo "   pip install mindspore-{version}-cp39-cp39-linux_aarch64.whl"
    echo ""
    exit 1
}

echo "下载完成，正在安装..."
pip3 install "$WHEEL_NAME" --user

echo ""
echo "======================================"
echo "安装验证"
echo "======================================"

python3 -c "import mindspore; print(f'MindSpore 版本: {mindspore.__version__}')" || {
    echo ""
    echo "安装可能有问题，请检查..."
    exit 1
}

echo ""
echo "✓ MindSpore 安装成功！"
echo ""
echo "======================================"
echo "后续步骤"
echo "======================================"
echo ""
echo "1. 安装 transformers (用于加载 HuggingFace 模型):"
echo "   pip install transformers safetensors modelscope"
echo ""
echo "2. 运行模型下载:"
echo "   python3 download_qwen_model.py"
echo ""
echo "3. 运行推理测试:"
echo "   python3 infer_qwen_lite.py"
echo ""
