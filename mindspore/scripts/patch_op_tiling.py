#!/usr/bin/env python3
"""
CANN 7.1.0 op_tiling.py 修复脚本

问题: sys_version 变量未定义导致 NameError
原因: if 块内定义的变量在块外使用，但 if 条件不满足时变量未定义
"""

import os
import shutil

# 源文件路径
SOURCE_FILE = "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/utils/op_tiling.py"
BACKUP_FILE = SOURCE_FILE + ".cann_fix_backup"
PATCHED_FILE = "./op_tiling_patched.py"

print("="*60)
print("CANN 7.1.0 op_tiling.py 修复工具")
print("="*60)
print(f"源文件: {SOURCE_FILE}")
print(f"备份: {BACKUP_FILE}")
print()

# 检查权限
if not os.access(SOURCE_FILE, os.R_OK):
    print("错误: 无法读取源文件（需要权限）")
    exit(1)

# 创建本地备份
if not os.path.exists("./op_tiling.py.bak"):
    shutil.copy(SOURCE_FILE, "./op_tiling.py.bak")
    print("✓ 已创建本地备份: ./op_tiling.py.bak")

# 读取源文件
with open(SOURCE_FILE, 'r') as f:
    content = f.read()
    lines = content.split('\n')

# 查找并修复问题
# 问题代码: sys_version 只在 if 块内定义，但在块外使用
# 修复: 在 if 块之前初始化 sys_version

print("正在应用补丁...")

# 新代码：在 if os.path.exists(scene_info_path): 之前初始化
old_code = """tiling_cust_path = os.path.join("ai_core", "tbe", "op_tiling", "liboptiling.so")
#Get system info
if os.path.exists(scene_info_path):"""

new_code = """tiling_cust_path = os.path.join("ai_core", "tbe", "op_tiling", "liboptiling.so")
#Get system info
# Initialize sys_version with default value to fix CANN 7.1.0 bug
sys_version = "linux"  # Default OS version for Ascend platform
if os.path.exists(scene_info_path):"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✓ 已应用补丁: 添加 sys_version 默认值")
else:
    print("警告: 未找到目标代码，可能已修复或版本不同")
    print("尝试手动修复...")

    # 备选方案：直接查找 sys_version 使用位置并在前面添加初始化
    for i, line in enumerate(lines):
        if 'tiling_so_arch_path = os.path.join("ai_core", "tbe", "op_tiling", "lib", sys_version,' in line:
            # 在这行前面插入 sys_version 初始化
            # 找到最近的空行或 # 注释行后插入
            insert_pos = i
            for j in range(i-1, max(0, i-10), -1):
                if lines[j].strip() == '' or lines[j].strip().startswith('#'):
                    insert_pos = j + 1
                    break
            lines.insert(insert_pos, "# Fix CANN 7.1.0 bug: initialize sys_version")
            lines.insert(insert_pos + 1, "sys_version = \"linux\"  # Default OS version")
            print(f"✓ 已在行 {insert_pos} 插入 sys_version 初始化")
            break

    content = '\n'.join(lines)

# 写入修复后的文件
with open(PATCHED_FILE, 'w') as f:
    f.write(content)

print()
print("✓ 修复后的文件已保存: ", PATCHED_FILE)
print()
print("="*60)
print("应用修复")
print("="*60)
print()
print("请运行以下命令应用修复（需要 sudo 权限）：")
print()
print(f"  sudo cp {PATCHED_FILE} {SOURCE_FILE}")
print()
print("或者运行：")
print()
print("  sudo python3 -c \"import shutil; shutil.copy('{}', '{}')\"".format(PATCHED_FILE, SOURCE_FILE))
print()
print("验证修复:")
print("  source /usr/local/Ascend/ascend-toolkit/set_env.sh")
print("  python3 -c 'import mindspore; mindspore.run_check()'")
