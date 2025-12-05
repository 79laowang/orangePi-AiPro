#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACL 最小示例 - 核心步骤演示
展示使用 ACL 的最少必要代码
"""

import acl
import numpy as np

# 配置
MODEL_PATH = "resnet50.om"
DEVICE_ID = 0

def minimal_acl_example():
    """最小化的 ACL 使用流程"""

    print("=== ACL 最小示例 ===\n")

    # 步骤 1: 初始化 ACL
    print("步骤 1: acl.init() - 初始化 ACL 运行时")
    ret = acl.init()
    assert ret == acl.ACL_SUCCESS, f"ACL 初始化失败: {ret}"
    print("✓ ACL 初始化成功\n")

    # 步骤 2: 设置设备
    print("步骤 2: acl.rt.set_device() - 设置 NPU 设备")
    ret = acl.rt.set_device(DEVICE_ID)
    assert ret == acl.ACL_SUCCESS, f"设置设备失败: {ret}"
    print(f"✓ 已设置设备 {DEVICE_ID}\n")

    # 步骤 3: 创建执行环境
    print("步骤 3: 创建 Context 和 Stream")
    context, ret = acl.rt.create_context(DEVICE_ID)
    assert ret == acl.ACL_SUCCESS, f"创建上下文失败: {ret}"
    stream, ret = acl.rt.create_stream(context)
    assert ret == acl.ACL_SUCCESS, f"创建流失败: {ret}"
    print("✓ 执行环境创建完成\n")

    # 步骤 4: 加载模型
    print("步骤 4: acl.mdl.load_from_file() - 加载 .om 模型")
    model_id, model_desc = acl.mdl.load_from_file(MODEL_PATH)
    assert model_id is not None, "模型加载失败"
    print(f"✓ 模型加载成功 (ID: {model_id})\n")

    # 步骤 5: 获取模型信息
    print("步骤 5: 获取模型输入输出信息")
    input_num = acl.mdl.get_num_inputs(model_desc)
    output_num = acl.mdl.get_num_outputs(model_desc)
    print(f"输入数量: {input_num}")
    print(f"输出数量: {output_num}")

    # 步骤 6: 分配内存
    print("\n步骤 6: 分配输入输出内存")
    input_buffers = []
    output_buffers = []

    # 为每个输入分配内存
    for i in range(input_num):
        size = acl.mdl.get_input_size_by_index(model_desc, i)
        buffer, ret = acl.rt.malloc(size, DEVICE_ID)
        assert ret == acl.ACL_SUCCESS, f"分配输入内存失败: {ret}"
        input_buffers.append(buffer)
        print(f"  输入 {i}: 分配 {size} bytes")

    # 为每个输出分配内存
    for i in range(output_num):
        size = acl.mdl.get_output_size_by_index(model_desc, i)
        buffer, ret = acl.rt.malloc(size, DEVICE_ID)
        assert ret == acl.ACL_SUCCESS, f"分配输出内存失败: {ret}"
        output_buffers.append(buffer)
        print(f"  输出 {i}: 分配 {size} bytes")

    print("✓ 内存分配完成\n")

    # 步骤 7: 准备输入数据 (模拟)
    print("步骤 7: 准备输入数据")
    print("  这里通常是预处理后的图像数据")
    print("  例如: (1, 3, 224, 224) 的 numpy 数组")
    print("✓ 输入数据准备就绪\n")

    # 步骤 8: 创建数据集
    print("步骤 8: 创建输入输出数据集")

    # 创建输入数据集
    input_dataset = acl.mdl.create_dataset()
    for buffer in input_buffers:
        data_item = acl.create_data_buffer(buffer)
        acl.mdl.add_dataset_tensor(input_dataset, acl.MDL_INPUT, data_item)

    # 创建输出数据集
    output_dataset = acl.mdl.create_dataset()
    for buffer in output_buffers:
        data_item = acl.create_data_buffer(buffer)
        acl.mdl.add_dataset_tensor(output_dataset, acl.MDL_OUTPUT, data_item)

    print("✓ 数据集创建完成\n")

    # 步骤 9: 执行推理
    print("步骤 9: acl.mdl.execute() - 执行模型推理")
    print("  这是最关键的一步，模型在 NPU 上运行")
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
    assert ret == acl.ACL_SUCCESS, f"推理执行失败: {ret}"
    print("✓ 推理完成\n")

    # 步骤 10: 获取结果
    print("步骤 10: 获取推理结果")
    print("  结果现在在 output_buffers 中的设备内存上")
    print("  需要使用 memcpy 复制到主机内存\n")

    # 步骤 11: 清理资源
    print("步骤 11: 清理资源")
    acl.mdl.destroy_dataset(input_dataset)
    acl.mdl.destroy_dataset(output_dataset)

    for buffer in input_buffers + output_buffers:
        acl.rt.free(buffer)

    acl.mdl.destroy_model(model_id)
    acl.mdl.destroy_desc(model_desc)
    acl.rt.destroy_stream(stream)
    acl.rt.destroy_context(context)
    acl.rt.reset_device(DEVICE_ID)

    print("✓ 资源清理完成\n")

    print("=== 示例完成 ===")


if __name__ == "__main__":
    try:
        minimal_acl_example()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print(f"错误类型: {type(e).__name__}")
