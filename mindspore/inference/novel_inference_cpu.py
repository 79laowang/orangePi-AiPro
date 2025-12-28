#!/usr/bin/env python3
"""
中文小说创作 - CPU推理模式
使用 Qwen2-1.5B-Instruct 预训练模型，无需训练
"""

import mindspore
import numpy as np
from mindspore import Tensor
import time

print("=" * 60)
print("中文小说创作助手 - CPU模式")
print("=" * 60)

# 检查 MindSpore
try:
    print(f"\n✓ MindSpore 版本: {mindspore.__version__}")
    mindspore.set_context(device_target="CPU", mode=mindspore.PYNATIVE_MODE)
    print("✓ 设备模式: CPU (PYNATIVE)")
except Exception as e:
    print(f"✗ MindSpore 初始化失败: {e}")
    exit(1)

# 方案 A: 使用 transformers + CPU (推荐)
print("\n" + "=" * 60)
print("方案 A: 使用 transformers (推荐)")
print("=" * 60)
print("""
优点:
- 直接使用 HuggingFace 生态
- CPU 模式运行稳定
- 支持流式生成
- 内存占用低

安装:
    pip install transformers torch sentencepiece

使用示例:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

    prompt = "写一段关于江湖剑客的武侠小说开头："
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    result = tokenizer.decode(outputs[0])
""")

# 方案 B: 使用 MindSpore CPU 模式
print("\n" + "=" * 60)
print("方案 B: 使用 MindSpore (需要转换模型)")
print("=" * 60)
print("""
优点:
- 原生 MindSpore 支持
- 可以使用 MindSpore 优化

缺点:
- 需要转换 HuggingFace 模型
- 文档较少
- CPU 模式可能不如 transformers 直接

步骤:
1. 下载 Qwen2-1.5B-Instruct 模型
2. 转换为 MindSpore 格式
3. 使用 CPU 推理
""")

# 方案 C: 使用更小的模型
print("\n" + "=" * 60)
print("方案 C: 使用更小模型 (Qwen2-0.5B)")
print("=" * 60)
print("""
优点:
- 内存占用更低 (~3GB)
- 速度更快
- 同样 CPU 模式运行

模型:
- Qwen2-0.5B-Instruct
- 适合中文对话和创作
""")

# 小说创作示例 Prompt
print("\n" + "=" * 60)
print("小说创作 Prompt 示例")
print("=" * 60)

prompts = {
    "武侠": """请写一段武侠小说开头：

背景：明朝万历年间，江南水乡
主角：年轻剑客，身怀绝世剑法
情节：初入江湖，卷入一场阴谋

要求：
- 文笔优美，古风韵味
- 环境描写生动
- 引人入胜的悬念""",

    "仙侠": """请写一段仙侠小说开头：

背景：修仙世界，灵气复苏
主角：废柴少年获得神秘传承
情节：踏上逆天改命的修仙之路

要求：
- 修炼体系清晰
- 想象力丰富
- 节奏明快""",

    "都市": """请写一段都市小说开头：

背景：现代都市，隐藏的超凡世界
主角：普通大学生觉醒异能
情节：发现世界真相，加入神秘组织

要求：
- 现代感强
- 悬念设置巧妙
- 人物性格鲜明"""
}

for genre, prompt in prompts.items():
    print(f"\n【{genre}风格】")
    print("-" * 40)
    print(prompt[:100] + "...")

print("\n" + "=" * 60)
print("推荐流程")
print("=" * 60)
print("""
1. 安装依赖:
   pip install transformers torch sentencepiece

2. 运行推理脚本:
   python3 infer_qwen_cpu.py

3. 交互式创作:
   - 输入小说类型
   - 设置角色和情节
   - 流式生成内容

4. 后续优化 (可选):
   - 收集生成内容
   - 整理成数据集
   - 使用 LoRA 微调（需要更多资源）
""")

print("\n" + "=" * 60)
print("是否创建 CPU 推理脚本？")
print("=" * 60)
