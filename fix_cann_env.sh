#!/bin/bash
# CANN 环境变量修复脚本
# 修复 ASCEND_OPP_PATH 路径错误问题

echo "修复 CANN 环境变量..."

# 移除错误的 ASCEND_OPP_PATH
unset ASCEND_OPP_PATH

# 设置正确的路径
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=${ASCEND_HOME}/opp
export LD_LIBRARY_PATH=${ASCEND_HOME}/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=${ASCEND_HOME}/python/site-packages:${PYTHONPATH}

echo "环境变量已修复:"
echo "  ASCEND_HOME=${ASCEND_HOME}"
echo "  ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
echo ""
echo "验证修复..."
source ${ASCEND_HOME}/set_env.sh

python3 -c "
import mindspore
print('MindSpore 版本:', mindspore.__version__)
mindspore.set_context(device_target='CPU')
import mindspore.ops as ops
import numpy as np
x = mindspore.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = mindspore.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
result = ops.add(x, y)
print('CPU 推理测试: 成功')
"
