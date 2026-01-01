# Qwen Chatbot for Orange Pi AI Pro

基于香橙派AIpro + MindSpore 实现的通义千问Qwen聊天机器人。

## 项目简介

本项目在Orange Pi AIPro开发板上基于MindSpore框架实现Qwen-1.5-0.5B模型，体验和模型的对话互动，完成和聊天机器人对话。

基于 [orange-pi-mindspore](https://github.com/mindspore-lab/orange-pi-mindspore) 参考实现。

- **模型**: Qwen-1.5-0.5B-Chat (500M参数)
- **框架**: MindSpore 2.6.0 + mindnlp 0.4.1
- **硬件**: Orange Pi AI Pro (Ascend 310/310B NPU)

## 环境要求

### 硬件
- 香橙派 AIpro 16G 8-12T 开发板
- **NPU**: Ascend 310B4
- **内存**: 15GB 总计，推荐至少 4GB 可用 RAM

### 软件
- **镜像**: Ubuntu 22.04.3 LTS (Kernel 5.10.0+)
- **CANN**: 8.1.RC1 (Version 7.7.0.1.238)
- **npu-smi**: 25.2.0
- **MindSpore**: 2.6.0
- **mindnlp**: 0.4.1
- **Python**: 3.10+

### 当前系统配置

本项目在以下配置中测试通过：

| 组件 | 版本/规格 |
|------|-----------|
| 硬件平台 | Orange Pi AI Pro |
| NPU | Ascend 310B4 |
| 内存 | 15GB |
| 操作系统 | Ubuntu 22.04.3 LTS |
| 内核版本 | 5.10.0+ |
| CANN | 8.1.RC1 (7.7.0.1.238) |
| npu-smi | 25.2.0 |
| MindSpore | 2.6.0 |
| mindnlp | 0.4.1 |
| Gradio | 6.x |

### 环境准备参考

请参考昇思MindSpore官网--香橙派开发专区完成环境搭建:

1. [镜像烧录](https://www.mindspore.cn/docs/zh-CN/r2.6.0/orange_pi/environment_setup.html)
2. [CANN升级](https://www.mindspore.cn/docs/zh-CN/r2.6.0/orange_pi/environment_setup.html)
3. [MindSpore升级](https://www.mindspore.cn/docs/zh-CN/r2.6.0/orange_pi/environment_setup.html)

## 安装

### 1. 安装依赖

```bash
# 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import mindspore; print(mindspore.__version__)"
python -c "import mindnlp; print(mindnlp.__version__)"
python -c "import gradio; print(gradio.__version__)"
```

## 使用方法

### 模型下载 (中国用户)

首次运行前，建议设置 HuggingFace 镜像以加速下载:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 启动聊天界面

```bash
# 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 运行应用
python app.py
```

然后在浏览器中打开: http://127.0.0.1:7860/

### 命令行选项

```bash
# 使用默认设置
python app.py

# 创建公共链接 (用于外网访问测试)
python app.py --share

# 指定端口
python app.py --server-port 8080

# 指定服务器地址
python app.py --server-name 0.0.0.0 --server-port 7860
```

## 项目结构

```
qwen-chat/
├── app.py              # Gradio 聊天界面
├── requirements.txt    # Python 依赖
└── README.md          # 本文件
```

## 功能特性

- ✅ 流式输出 - 实时显示生成过程
- ✅ 聊天历史 - 自动维护对话上下文
- ✅ 中英双语 - 支持中英文对话
- ✅ 示例问题 - 预设常用问题快速测试
- ✅ NPU 加速 - 利用 Ascend NPU 进行推理加速

## 预期性能

| 指标 | 性能 |
|------|------|
| 首次响应 | 1-2 分钟 (模型加载) |
| 后续响应 | 10-30 秒 |
| 内存占用 | 约 2-3 GB |

## 常见问题

### Q: 首次运行很慢？
A: 首次运行需要从 Hugging Face 下载模型，约 1GB 数据。下载后会缓存，后续启动会快很多。

### Q: 如何修改系统提示词？
A: 编辑 `app.py` 中的 `SYSTEM_PROMPT` 变量。

### Q: 可以使用其他 Qwen 模型吗？
A: 可以，修改 `app.py` 中的 `MODEL_NAME` 变量，例如:
```python
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
```
注意: 更大的模型需要更多内存。

### Q: 内存不足怎么办？
A: 可以尝试:
1. 关闭其他占用内存的程序
2. 使用更小的模型
3. 减少 `max_new_tokens` 参数
4. 清理缓存: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

### Q: NPU 错误如何处理？
A: 确保 CANN 环境正确设置:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
```

## 技术探索记录

### JIT 优化尝试

本项目最初尝试使用 MindSpore JIT 优化来提升推理性能，但遇到了以下问题：

1. **Repetition Penalty Bug**: mindnlp 0.4.1 的 `RepetitionPenaltyLogitsProcessor` 在 JIT 模式下存在 Int32/Int64 类型不匹配问题
2. **StaticCache 不兼容**: `model.jit()` + `StaticCache` 与 Qwen1.5-0.5B-Chat 模型的 attention 机制不兼容
3. **参考实现验证**: 经查阅 [orange-pi-mindspore](https://github.com/mindspore-lab/orange-pi-mindspore) 参考项目，其 Qwen1.5-0.5B 示例同样不使用 JIT 优化

**结论**: 对于 Qwen1.5-0.5B-Chat 模型，使用标准的 `model.generate()` 方法是当前最稳定的方案。JIT 优化在 DeepSeek-R1-Distill-Qwen-1.5B 等其他模型上可能有更好的支持。

### 遇到的错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ValueError: The mirror name not support` | 向 `from_pretrained()` 传递 `mirror=None` | 仅在 mirror 值非 None 时传递该参数 |
| `TypeError: All input must have the same data type!` | JIT 模式下 RepetitionPenaltyLogitsProcessor 类型不匹配 | 使用标准 model.generate() 不启用 JIT |
| `ValueError: Attention weights should be of size (...)` | `model.jit()` + StaticCache 与 Qwen1.5 不兼容 | 移除 model.jit() 调用 |
| `ValueError: too many values to unpack (expected 2)` | Gradio 6.x chat history 格式变更 | 使用兼容多种格式的历史解析函数 |
| `TypeError: can only concatenate str (not "list") to str` | `apply_chat_template` with `tokenize=True` 在消息格式异常时失败 | 确保 message content 为字符串，或使用 `tokenize=False` 后手动 tokenization |
| 模型输出 `{'text': '...', 'type': 'text'}` 结构化数据 | 模型异常输出格式，可能与聊天历史污染有关 | 简化实现，使用参考代码，避免复杂的输出处理 |
| 浏览器页面闪烁/崩溃 | 流式输出产生空 tokens 或无限循环 | 添加空 token 跳过和最大空 token 计数限制 |

### Gradio 6.x 兼容性问题

**问题**: Gradio 6.x 改变了 `ChatInterface` 中 `history` 参数的传递格式，从简单的 `(user_msg, ai_msg)` 元组列表变为更复杂的结构。

**解决方案**: 在 `build_input_from_chat_history` 函数中兼容多种格式：

```python
def build_input_from_chat_history(chat_history, msg: str):
    messages = [{'role': 'system', 'content': system_prompt}]
    for item in chat_history:
        if isinstance(item, tuple) and len(item) == 2:
            # Tuple format: (user_msg, ai_msg)
            user_msg, ai_msg = item
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
        elif len(item) == 2:
            # List format: [user_msg, ai_msg]
            user_msg, ai_msg = item
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
    messages.append({'role': 'user', 'content': msg})
    return messages
```

### 版本兼容性说明

本项目当前使用较新版本的软件栈，与参考文档有所不同：

| 组件 | 参考版本 | 当前版本 | 兼容性 |
|------|----------|----------|--------|
| CANN | 8.0.RC3.alpha002 | 8.1.RC1 | ✅ 向后兼容 |
| MindSpore | 2.4.10 | 2.6.0 | ✅ 向后兼容 |
| mindnlp | - | 0.4.1 | ✅ 正常工作 |
| Gradio | 5.x (推测) | 6.x | ⚠️ 需要 history 格式兼容处理 |

**建议**: 如遇到问题，可优先参考 [昇腾社区技术文章](https://www.hiascend.com/developer/techArticles/20250424-3) 中的原始实现。

## 参考资源

- [orange-pi-mindspore](https://github.com/mindspore-lab/orange-pi-mindspore) - 参考实现
- [昇腾社区 - 基于香橙派AIpro+MindSpore实现Qwen聊天机器人](https://www.hiascend.com/developer/techArticles/20250424-3)
- [Qwen Models on Hugging Face](https://huggingface.co/Qwen)
- [MindSpore 官网](https://www.mindspore.cn/)
- [MindNLP 文档](https://github.com/mindspore-lab/mindnlp)

## 许可证

本项目遵循 Apache 2.0 许可证。
