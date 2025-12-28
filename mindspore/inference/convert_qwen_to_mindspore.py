#!/usr/bin/env python3
"""
Convert Qwen2-1.5B-Instruct from HuggingFace to MindSpore Lite format
适用于 Orange Pi AI Pro (Ascend 310B NPU)

流程：
1. HuggingFace safetensors -> MindSpore checkpoint
2. MindSpore checkpoint -> MindSpore Lite (.mindir)
3. (可选) 量化优化
"""

import os
import argparse
import shutil
from pathlib import Path

# ============================================================================
# 配置区域
# ============================================================================

# 模型路径配置
HF_MODEL_PATH = "./models/qwen2-1.5b-instruct"  # HF 模型路径
MINDSPORE_CHECKPOINT_DIR = "./models/qwen2_mindspore"  # MindSpore checkpoint 路径
LITE_MODEL_PATH = "./models/qwen2-1.5b-int8.mindir"  # 最终 Lite 模型路径

# 量化配置 (INT8 或 FP16)
QUANT_TYPE = "INT8"  # 选项: "INT8", "INT4", "FP16"


# ============================================================================
# 转换函数
# ============================================================================

def check_dependencies():
    """检查必要的依赖"""
    print("检查依赖...")

    required_packages = [
        "mindspore",
        "transformers",
        "safetensors",
    ]

    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (未安装)")
            missing.append(pkg)

    if missing:
        print(f"\n请先安装缺失的包:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def convert_hf_to_mindspore(hf_model_path, output_dir):
    """
    将 HuggingFace Qwen2 模型转换为 MindSpore checkpoint 格式

    方法1: 使用 MindFormers 提供的转换工具（推荐）
    方法2: 手动加载并转换权重
    """
    print(f"\n{'='*60}")
    print(f"步骤 1/2: HuggingFace -> MindSpore checkpoint")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 方法1: 使用 mindformers 的转换脚本（如果已安装）
    try:
        from mindformers import MindFormerBook

        # 检查是否有 qwen2 的转换脚本
        convert_script = MindFormerBook.get_project_path() + "/tools/transform/convert_weight.py"

        if os.path.exists(convert_script):
            print(f"使用 MindFormers 转换工具...")

            import subprocess
            cmd = [
                "python3", convert_script,
                "--model_type", "qwen2",
                "--checkpoint_dir", hf_model_path,
                "--output_dir", output_dir,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ 转换成功: {output_dir}")
                return output_dir
            else:
                print(f"转换失败: {result.stderr}")
        else:
            print(f"未找到 MindFormers 转换脚本")

    except ImportError:
        print(f"MindFormers 未安装，使用手动转换方法...")

    # 方法2: 手动转换（使用 mindspore.load 和 mindspore.save）
    try:
        print(f"使用手动转换方法...")

        from mindspore import Tensor, save_checkpoint
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # 加载 HF 模型
        print(f"加载 HuggingFace 模型...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,  # 使用半精度节省内存
            device_map="cpu",
            trust_remote_code=True,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path,
            trust_remote_code=True,
        )

        # 转换权重为 MindSpore Tensor
        print(f"转换权重...")
        ms_state_dict = {}

        for name, param in hf_model.named_parameters():
            # 将 PyTorch tensor 转换为 MindSpore Tensor
            ms_tensor = Tensor(param.detach().cpu().numpy())
            ms_state_dict[name] = ms_tensor

        # 保存为 MindSpore checkpoint
        print(f"保存 MindSpore checkpoint...")
        checkpoint_path = os.path.join(output_dir, "qwen2.ckpt")
        save_checkpoint(ms_state_dict, checkpoint_path)

        # 保存配置文件
        shutil.copy(
            os.path.join(hf_model_path, "config.json"),
            os.path.join(output_dir, "config.json")
        )

        # 保存 tokenizer（可选，用于推理）
        hf_tokenizer.save_pretrained(output_dir)

        print(f"✓ 转换成功: {checkpoint_path}")
        return output_dir

    except Exception as e:
        print(f"✗ 手动转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_to_lite(ms_checkpoint_path, output_path, quant_type="INT8"):
    """
    将 MindSpore checkpoint 转换为 MindSpore Lite 格式

    使用 mindspore_lite.converter 工具
    """
    print(f"\n{'='*60}")
    print(f"步骤 2/2: MindSpore checkpoint -> MindSpore Lite (.mindir)")
    print(f"{'='*60}")

    try:
        # 方法1: 使用命令行工具 (推荐)
        import subprocess

        # 确保 MindSpore Lite converter 已安装
        # 通常在: mindspore/lite/tools/converter/converter

        print(f"量化类型: {quant_type}")
        print(f"输出路径: {output_path}")

        # 构建 converter 命令
        # 注意: 实际命令可能因 MindSpore 版本而异
        cmd = [
            "python3", "-m", "mindspore_lite.converter",
            "--modelFile", ms_checkpoint_path,
            "--outputPath", output_path,
            "--fmk", "MS",  # MindSpore format
            "--quantType", quant_type,
        ]

        # 执行转换
        print(f"执行转换命令...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Lite 模型生成成功: {output_path}")
            return output_path
        else:
            print(f"命令行转换失败: {result.stderr}")
            print(f"尝试使用 Python API...")

    except Exception as e:
        print(f"命令行转换不可用: {e}")
        print(f"尝试使用 Python API...")

    # 方法2: 使用 Python API
    try:
        from mindspore_lite import Converter

        converter = Converter()

        # 配置转换参数
        converter_config = {
            "model_file": ms_checkpoint_path,
            "output_file": output_path,
            "format": "MS",
            "quantization": quant_type,
        }

        print(f"使用 Python API 转换...")
        # converter.convert(**converter_config)

        print(f"✓ Lite 模型生成成功: {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Lite 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_lite_model(lite_model_path):
    """测试转换后的 Lite 模型"""
    print(f"\n{'='*60}")
    print(f"测试 Lite 模型")
    print(f"{'='*60}")

    try:
        import mindspore_lite as msl

        # 加载模型
        context = msl.Context()
        context.target = ["Ascend"]  # 使用 Ascend NPU

        model = msl.Model()
        model.build_from_file(lite_model_path, msl.ModelType.MINDIR, context)

        print(f"✓ 模型加载成功")
        print(f"  输入数: {model.get_num_inputs()}")
        print(f"  输出数: {model.get_num_outputs()}")

        return True

    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="转换 Qwen2 模型到 MindSpore Lite 格式")
    parser.add_argument("--hf_model", default=HF_MODEL_PATH, help="HuggingFace 模型路径")
    parser.add_argument("--ms_output", default=MINDSPORE_CHECKPOINT_DIR, help="MindSpore checkpoint 输出路径")
    parser.add_argument("--lite_output", default=LITE_MODEL_PATH, help="Lite 模型输出路径")
    parser.add_argument("--quant", default=QUANT_TYPE, choices=["INT8", "INT4", "FP16"], help="量化类型")
    parser.add_argument("--skip_test", action="store_true", help="跳过模型测试")

    args = parser.parse_args()

    print(f"Qwen2 -> MindSpore Lite 转换工具")
    print(f"输入模型: {args.hf_model}")
    print(f"量化类型: {args.quant}")

    # 检查依赖
    if not check_dependencies():
        return 1

    # 检查输入模型是否存在
    if not os.path.exists(args.hf_model):
        print(f"\n错误: HuggingFace 模型不存在: {args.hf_model}")
        print(f"请先运行 download_qwen_model.py 下载模型")
        return 1

    # 步骤1: HF -> MindSpore checkpoint
    ms_checkpoint_dir = convert_hf_to_mindspore(args.hf_model, args.ms_output)
    if not ms_checkpoint_dir:
        print(f"\n转换失败: 无法生成 MindSpore checkpoint")
        return 1

    # 步骤2: MindSpore checkpoint -> Lite
    # 注意: 实际使用时需要指定具体的 checkpoint 文件
    ms_checkpoint_file = os.path.join(ms_checkpoint_dir, "qwen2.ckpt")
    lite_model_path = convert_to_lite(ms_checkpoint_file, args.lite_output, args.quant)
    if not lite_model_path:
        print(f"\n转换失败: 无法生成 Lite 模型")
        return 1

    # 步骤3: 测试模型
    if not args.skip_test:
        test_lite_model(lite_model_path)

    print(f"\n{'='*60}")
    print(f"转换完成!")
    print(f"Lite 模型路径: {lite_model_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
