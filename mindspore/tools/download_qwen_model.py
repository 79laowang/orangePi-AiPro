#!/usr/bin/env python3
"""
Download Qwen2-1.5B-Instruct model from ModelScope
适用于 Orange Pi AI Pro (Ascend 310B)
"""

from modelscope import snapshot_download
import os

# 模型 ID
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

# 本地保存路径（可根据需要修改）
SAVE_DIR = "./models/qwen2-1.5b-instruct"

def download_model():
    """下载 Qwen2-1.5B-Instruct 模型"""

    print(f"正在下载模型: {MODEL_ID}")
    print(f"缓存目录: {os.path.abspath(SAVE_DIR)}")
    print("-" * 50)

    try:
        # 使用 modelscope 下载（不指定 cache_dir，使用默认缓存）
        model_dir = snapshot_download(
            MODEL_ID,
            revision="master",
        )

        # 创建软链接到固定路径，方便后续使用
        link_path = "./models/qwen2-1.5b-instruct"
        os.makedirs(os.path.dirname(link_path), exist_ok=True)

        # 删除旧链接（如果存在）
        if os.path.islink(link_path):
            os.remove(link_path)
        elif os.path.exists(link_path):
            print(f"警告: {link_path} 已存在且不是软链接")

        # 创建软链接
        os.symlink(model_dir, link_path)

        print("-" * 50)
        print(f"模型下载完成！")
        print(f"实际路径: {model_dir}")
        print(f"软链接: {link_path} -> {model_dir}")

        # 列出下载的文件
        print("\n下载的文件:")
        for item in sorted(os.listdir(model_dir)):
            item_path = os.path.join(model_dir, item)
            if os.path.isfile(item_path):
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"  - {item} ({size_mb:.2f} MB)")
            elif os.path.isdir(item_path):
                print(f"  - {item}/ (目录)")

        return link_path  # 返回软链接路径

    except Exception as e:
        print(f"下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 确保目标目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_path = download_model()

    if model_path:
        print(f"\n提示: 可以使用以下路径加载模型:")
        print(f"  model_path = \"{model_path}\"")
