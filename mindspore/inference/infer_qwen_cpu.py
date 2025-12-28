#!/usr/bin/env python3
"""
中文小说创作 - CPU 推理脚本
使用 Qwen2-1.5B-Instruct 预训练模型
无需训练，直接生成小说内容
"""

import sys
import time

print("=" * 60)
print("中文小说创作助手")
print("=" * 60)

# 检查依赖
try:
    import transformers
    import torch
    print(f"\n✓ transformers 版本: {transformers.__version__}")
    print(f"✓ torch 版本: {torch.__version__}")
    print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
    print(f"✓ 使用设备: {'CPU' if not torch.cuda.is_available() else 'CUDA'}")
except ImportError as e:
    print(f"\n✗ 缺少依赖，请安装:")
    print("  pip install transformers torch sentencepiece")
    sys.exit(1)

# 加载模型
print("\n" + "=" * 60)
print("加载模型...")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen2-1.5B-Instruct"
# 如果已下载到本地，使用本地路径
# MODEL_PATH = "./models/qwen2-1.5b-instruct"

print(f"模型路径: {MODEL_PATH}")
print("正在加载... (首次运行需要下载模型，约 3GB)")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,  # 使用半精度节省内存
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    print("\n提示: 首次运行需要从 ModelScope 下载模型:")
    print("  python3 download_qwen_model.py")
    sys.exit(1)

# 小说创作预设模板
NOVEL_TEMPLATES = {
    "武侠": {
        "system": "你是一位擅长创作武侠小说的作家，文笔优美，擅长描写江湖恩怨和武功招式。",
        "examples": [
            "请写一段武侠小说开头：\n背景：明朝万历年间，江南水乡\n主角：年轻剑客，身怀绝世剑法\n情节：初入江湖，卷入一场阴谋",
            "续写一段剑客比武的场景：\n两个高手在雨中对峙，剑气纵横",
            "描写一位武林前辈的出场：\n白发老者，仙风道骨，深不可测"
        ]
    },
    "仙侠": {
        "system": "你是一位擅长创作仙侠小说的作家，想象力丰富，擅长构建修仙体系和世界观。",
        "examples": [
            "请写一段仙侠小说开头：\n背景：修仙世界，灵气复苏\n主角：废柴少年获得神秘传承\n情节：踏上逆天改命的修仙之路",
            "描写主角突破境界的场景：\n雷劫降临，天威浩荡",
            "设计一个修仙宗门的场景：\n云雾缭绕的仙山，弟子练功"
        ]
    },
    "都市": {
        "system": "你是一位擅长创作都市小说的作家，擅长描写现代生活和社会关系。",
        "examples": [
            "请写一段都市小说开头：\n背景：现代都市，隐藏的超凡世界\n主角：普通大学生觉醒异能\n情节：发现世界真相，加入神秘组织",
            "描写主角第一次使用异能的场景：\n时间突然静止，周围一切定格",
            "写一段都市职场的剧情：\n办公室里的勾心斗角"
        ]
    }
}

def generate_novel(prompt, max_length=500, temperature=0.9, top_p=0.95):
    """生成小说内容"""
    messages = [
        {"role": "system", "content": "你是一位专业的小说作家，擅长创作引人入胜的故事。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"\n生成中... (max_length={max_length}, temperature={temperature})")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start_time

    generated_ids = outputs[0][len(inputs[0]):]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"✓ 生成完成，耗时 {elapsed:.1f} 秒")
    print(f"✓ 生成了 {len(result)} 个字符")

    return result

def interactive_mode():
    """交互式创作模式"""
    print("\n" + "=" * 60)
    print("交互式小说创作模式")
    print("=" * 60)

    while True:
        print("\n【选择小说类型】")
        for i, genre in enumerate(Novel_TEMPLATES.keys(), 1):
            print(f"  {i}. {genre}")
        print("  0. 退出")

        choice = input("\n请选择 (0-{}): ".format(len(NOVEL_TEMPLATES)))
        if choice == "0":
            print("再见！")
            break

        try:
            genre_idx = int(choice) - 1
            genre = list(NOVEL_TEMPLATES.keys())[genre_idx]
            template = NOVEL_TEMPLATES[genre]
        except (ValueError, IndexError):
            print("无效选择")
            continue

        print(f"\n【{genre}风格】")
        print("-" * 40)

        # 显示示例
        print("示例 prompt:")
        for i, example in enumerate(template['examples'], 1):
            print(f"  {i}. {example[:50]}...")

        print("\n选择:")
        print("  1-3. 使用示例 prompt")
        print("  4. 自定义 prompt")

        sub_choice = input("请选择: ")

        if sub_choice in ["1", "2", "3"]:
            prompt = template['examples'][int(sub_choice) - 1]
        elif sub_choice == "4":
            prompt = input("请输入你的 prompt: ")
        else:
            print("无效选择")
            continue

        # 参数调整
        print("\n参数设置 (直接回车使用默认值):")
        max_length = input("生成长度 (默认500): ") or "500"
        temperature = input("创意程度 (0.7-1.2, 默认0.9): ") or "0.9"

        # 生成
        print("\n" + "=" * 60)
        result = generate_novel(
            prompt,
            max_length=int(max_length),
            temperature=float(temperature)
        )

        print("\n【生成结果】")
        print("-" * 60)
        print(result)
        print("-" * 60)

        # 保存
        save = input("\n是否保存到文件? (y/n): ")
        if save.lower() == "y":
            filename = f"novel_{genre}_{int(time.time())}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Generated:\n{result}")
            print(f"✓ 已保存到 {filename}")

def demo_mode():
    """演示模式 - 快速示例"""
    print("\n" + "=" * 60)
    print("演示模式 - 武侠小说开头")
    print("=" * 60)

    prompt = """请写一段武侠小说开头：

背景：明朝万历年间，江南水乡，烟雨朦胧
主角：年轻剑客，身怀绝世剑法"流云剑"
情节：初入江湖，来到一家客栈，却不知已卷入一场江湖阴谋

要求：
- 文笔优美，古风韵味
- 环境描写生动
- 引人入胜的悬念
- 500字左右"""

    print(f"\nPrompt:\n{prompt}\n")

    result = generate_novel(prompt, max_length=800, temperature=0.9)

    print("\n【生成结果】")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # 保存
    filename = f"novel_demo_{int(time.time())}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Generated:\n{result}")
    print(f"\n✓ 已保存到 {filename}")

if __name__ == "__main__":
    print("\n选择模式:")
    print("  1. 演示模式 (快速体验)")
    print("  2. 交互模式 (自由创作)")

    mode = input("\n请选择 (1/2): ")

    if mode == "1":
        demo_mode()
    elif mode == "2":
        interactive_mode()
    else:
        print("运行演示模式...")
        demo_mode()
