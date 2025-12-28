#!/usr/bin/env python3
"""
MindSpore Lite 推理脚本 - Qwen2-1.5B for 中文小说创作
适用于 Orange Pi AI Pro (Ascend 310B NPU)
"""

import mindspore_lite as msl
import numpy as np
from pathlib import Path


# ============================================================================
# 配置
# ============================================================================

MODEL_PATH = "./models/qwen2-1.5b-int8.mindir"

# 生成参数
MAX_LENGTH = 2048          # 最大生成长度
TEMPERATURE = 0.8          # 温度（越高越随机）
TOP_P = 0.9                # nucleus sampling
TOP_K = 50                 # top-k sampling


# ============================================================================
# 推理类
# ============================================================================

class Qwen2LiteInference:
    """Qwen2 MindSpore Lite 推理类"""

    def __init__(self, model_path: str):
        """
        初始化推理引擎

        Args:
            model_path: MindSpore Lite 模型路径 (.mindir)
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请先运行 convert_qwen_to_mindspore.py 转换模型"
            )

        # 配置 Ascend NPU 上下文
        self.context = msl.Context()
        self.context.target = ["Ascend"]
        self.context.precision = "fp16"  # 使用半精度加速

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = msl.Model()
        self.model.build_from_file(
            str(self.model_path),
            msl.ModelType.MINDIR,
            self.context
        )
        print(f"✓ 模型加载成功")

        # 获取输入输出信息
        self.num_inputs = self.model.get_num_inputs()
        self.num_outputs = self.model.get_num_outputs()

        print(f"  输入数: {self.num_inputs}")
        print(f"  输出数: {self.num_outputs}")

        # TODO: 初始化 tokenizer
        # tokenizer 需要从原始模型目录加载
        # self.tokenizer = ...

    def generate(
        self,
        prompt: str,
        max_length: int = MAX_LENGTH,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示词
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数

        Returns:
            生成的文本
        """
        # TODO: 实现 tokenization -> inference -> detokenization 流程
        #
        # 伪代码:
        # input_ids = self.tokenizer.encode(prompt)
        #
        # for _ in range(max_length):
        #     # 准备输入
        #     inputs = self._prepare_inputs(input_ids)
        #
        #     # 推理
        #     outputs = self.model.predict(inputs)
        #
        #     # 采样下一个 token
        #     next_token = self._sample(outputs, temperature, top_p, top_k)
        #
        #     input_ids.append(next_token)
        #
        #     # 检查结束条件
        #     if next_token == self.tokenizer.eos_token_id:
        #         break
        #
        # return self.tokenizer.decode(input_ids)

        raise NotImplementedError("请根据实际模型格式完善推理逻辑")

    def _prepare_inputs(self, input_ids):
        """准备模型输入"""
        # TODO: 根据模型要求准备输入张量
        # 通常包括: input_ids, attention_mask, position_ids 等
        pass

    def _sample(self, logits, temperature, top_p, top_k):
        """从 logits 中采样下一个 token"""
        # TODO: 实现温度采样 + top-p/top-k 过滤
        pass


# ============================================================================
# 中文小说创作辅助函数
# ============================================================================

NOVEL_WRITING_PROMPTS = {
    "outline": "请帮我根据以下主题写一个小说大纲：",
    "chapter": "请根据以下大纲写一章内容：",
    "dialogue": "请为以下场景写一段对话：",
    "description": "请描述以下场景：",
    "continue": "请继续写下去：",
}


def write_novel_scene(
    model: Qwen2LiteInference,
    scene_type: str,
    content: str,
    **kwargs
) -> str:
    """
    使用 AI 辅助创作小说场景

    Args:
        model: 推理模型
        scene_type: 场景类型 (outline, chapter, dialogue, description, continue)
        content: 创作内容/上下文
        **kwargs: 其他生成参数

    Returns:
        生成的内容
    """
    prompt_template = NOVEL_WRITING_PROMPTS.get(scene_type, "")

    if not prompt_template:
        print(f"未知场景类型: {scene_type}")
        return ""

    full_prompt = f"{prompt_template}\n\n{content}"

    print(f"生成中... (场景类型: {scene_type})")
    result = model.generate(full_prompt, **kwargs)

    return result


# ============================================================================
# 交互式创作界面
# ============================================================================

def interactive_writing():
    """交互式小说创作界面"""

    print("="*60)
    print("Qwen2 小说创作助手 (MindSpore Lite)")
    print("="*60)

    # 初始化模型
    try:
        model = Qwen2LiteInference(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    print("\n可用场景类型:")
    for key, desc in NOVEL_WRITING_PROMPTS.items():
        print(f"  - {key}: {desc}")

    print("\n命令:")
    print("  /quit - 退出")
    print("  /help - 显示帮助")

    # 主循环
    current_story = ""

    while True:
        user_input = input("\n> ").strip()

        if not user_input:
            continue

        if user_input == "/quit":
            print("再见！")
            break

        if user_input == "/help":
            # TODO: 显示详细帮助
            continue

        # 处理创作请求
        # TODO: 解析用户输入，调用模型生成
        print(f"（推理功能待完善）")


# ============================================================================
# 示例：批量生成
# ============================================================================

def example_batch_generation():
    """批量生成示例"""

    examples = [
        {
            "type": "outline",
            "content": "一个关于修仙世界的武侠故事",
        },
        {
            "type": "chapter",
            "content": "第一章：少年踏入修仙门",
        },
    ]

    model = Qwen2LiteInference(MODEL_PATH)

    for ex in examples:
        result = write_novel_scene(
            model,
            ex["type"],
            ex["content"],
            max_length=500,
        )
        print(f"\n{result}")
        print("-" * 40)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2 MindSpore Lite 推理")
    parser.add_argument("--model", default=MODEL_PATH, help="模型路径")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--example", action="store_true", help="运行示例")

    args = parser.parse_args()

    if args.interactive:
        interactive_writing()
    elif args.example:
        example_batch_generation()
    else:
        print("请使用 --interactive 或 --example 参数")
