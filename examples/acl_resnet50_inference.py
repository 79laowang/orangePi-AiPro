#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50 推理示例 - 使用 Ascend ACL (Ascend Computing Language)
适用于 Orange Pi AI Pro + Ascend 310/310B NPU

依赖: Python 3.8+, CANN, ACL
"""

import acl
import numpy as np
import cv2
import os
from pathlib import Path

class ResNet50ACLInference:
    """基于 ACL 的 ResNet50 模型推理类"""

    def __init__(self, model_path, device_id=0):
        """
        初始化 ACL 推理环境

        Args:
            model_path: .om 模型文件路径
            device_id: Ascend NPU 设备ID (默认为0)
        """
        self.model_path = model_path
        self.device_id = device_id
        self.model_desc = None
        self.model_id = None
        self.context = None
        self.stream = None
        self.input_buffers = []
        self.input_sizes = []
        self.output_buffers = []
        self.output_sizes = []

    def init_acl(self):
        """
        步骤1: 初始化 ACL 运行时环境
        这是使用 ACL 的第一步，必须在调用任何其他 ACL API 之前调用
        """
        print("🔧 初始化 ACL 运行时环境...")

        # 初始化 ACL 库
        # 参数说明:
        # - None: 使用默认配置
        # - 1: 单线程模式，简化错误处理
        ret = acl.init(None, 1)
        if ret != acl.ACL_SUCCESS:
            raise RuntimeError(f"ACL 初始化失败: {ret}")

        # 设置设备上下文 - 指定使用哪个 Ascend NPU 设备
        ret = acl.rt.set_device(self.device_id)
        if ret != acl.ACL_SUCCESS:
            raise RuntimeError(f"设置设备 {self.device_id} 失败: {ret}")

        # 创建执行上下文 (Context) 和执行流 (Stream)
        # Context: 管理设备内存和执行器
        # Stream: 异步执行队列，确保操作顺序
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != acl.ACL_SUCCESS:
            raise RuntimeError(f"创建上下文失败: {ret}")

        self.stream, ret = acl.rt.create_stream(self.context)
        if ret != acl.ACL_SUCCESS:
            raise RuntimeError(f"创建流失败: {ret}")

        print("✅ ACL 环境初始化完成")

    def load_model(self):
        """
        步骤2: 加载 .om 模型文件
        .om (Offline Model) 是 Ascend 平台的离线模型格式
        由 ATC (Ascend Tensor Compiler) 工具将 .pb/.onnx 等模型转换而来
        """
        print(f"📦 加载模型: {self.model_path}")

        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 从文件加载模型到内存
        # 返回值: (model_id, model_desc)
        # - model_id: 模型在设备内存中的标识符
        # - model_desc: 模型描述符，包含输入输出信息
        self.model_id, self.model_desc = acl.mdl.load_from_file(self.model_path)
        if ret != acl.ACL_SUCCESS:
            raise RuntimeError(f"加载模型失败: {ret}")

        print("✅ 模型加载成功")

        # 获取模型输入输出信息
        self._get_model_io_info()

    def _get_model_io_info(self):
        """
        获取模型的输入输出张量信息
        为后续的内存分配做准备
        """
        # 获取输入数量
        input_num = acl.mdl.get_num_inputs(self.model_desc)
        print(f"📊 模型输入数量: {input_num}")

        # 获取输出数量
        output_num = acl.mdl.get_num_outputs(self.model_desc)
        print(f"📊 模型输出数量: {output_num}")

        # 获取每个输入的形状和数据类型
        for i in range(input_num):
            # 获取输入张量的维度信息
            dims, ret = acl.mdl.get_input_dims(self.model_desc, i)
            if ret != acl.ACL_SUCCESS:
                raise RuntimeError(f"获取输入 {i} 维度失败: {ret}")

            # 获取输入张量的数据类型
            dtype = acl.mdl.get_input_data_type(self.model_desc, i)

            print(f"  输入 {i}: 形状={dims}, 数据类型={dtype}")

            # 计算输入张量的总大小 (字节)
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            self.input_sizes.append(size)

        # 获取每个输出的信息
        for i in range(output_num):
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            if ret != acl.ACL_SUCCESS:
                raise RuntimeError(f"获取输出 {i} 维度失败: {ret}")

            dtype = acl.mdl.get_output_data_type(self.model_desc, i)
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)

            print(f"  输出 {i}: 形状={dims}, 数据类型={dtype}, 大小={size} bytes")
            self.output_sizes.append(size)

    def allocate_buffers(self):
        """
        步骤3: 分配输入输出内存缓冲区
        在 NPU 上需要显式管理设备内存
        """
        print("💾 分配内存缓冲区...")

        # 为输入分配设备内存
        input_num = acl.mdl.get_num_inputs(self.model_desc)
        for i in range(input_num):
            # 在设备上分配指定大小的内存
            buffer, ret = acl.rt.malloc(self.input_sizes[i], self.device_id)
            if ret != acl.ACL_SUCCESS:
                raise RuntimeError(f"分配输入缓冲区 {i} 失败: {ret}")
            self.input_buffers.append(buffer)

        # 为输出分配设备内存
        output_num = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(output_num):
            buffer, ret = acl.rt.malloc(self.output_sizes[i], self.device_id)
            if ret != acl.ACL_SUCCESS:
                raise RuntimeError(f"分配输出缓冲区 {i} 失败: {ret}")
            self.output_buffers.append(buffer)

        print("✅ 内存缓冲区分配完成")

    def preprocess_image(self, image_path):
        """
        预处理输入图像
        ResNet50 标准预处理: 224x224, RGB, 归一化

        Args:
            image_path: 图像文件路径

        Returns:
            numpy.ndarray: 预处理后的图像数组 (1, 224, 224, 3)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # BGR -> RGB (OpenCV 默认 BGR, ResNet50 需要 RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整到 224x224
        image = cv2.resize(image, (224, 224))

        # 转换为浮点并归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0

        # 标准化 (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # 添加 batch 维度
        image = np.expand_dims(image, axis=0)

        return image

    def copy_input_to_device(self, image):
        """
        步骤4: 将输入数据从主机复制到设备 (NPU) 内存

        Args:
            image: numpy 数组格式的输入图像
        """
        # 将 numpy 数组转换为设备内存
        # 从主机内存复制到设备内存
        input_data = image.tobytes()
        ret = acl.rt.memcpy(
            self.input_buffers[0],  # 目标设备缓冲区
            self.input_sizes[0],    # 复制大小
            input_data,             # 源主机内存
            len(input_data),        # 源大小
            acl.MEMCPY_HOST_TO_DEVICE  # 复制方向: 主机到设备
        )
        if ret != acl.ACL_SUCCESS:
            raise RuntimeError(f"复制输入数据到设备失败: {ret}")

    def execute_inference(self):
        """
        步骤5: 执行模型推理

        执行步骤:
        1. 设置输入缓冲区
        2. 设置输出缓冲区
        3. 执行推理 (同步或异步)
        4. 获取结果
        """
        # 设置输入数据集
        # 将内存缓冲区绑定到输入张量
        input_dataset = acl.mdl.create_dataset()
        for buffer in self.input_buffers:
            # 为每个输入创建数据项 (DataItem)
            data_item = acl.create_data_buffer(buffer)
            acl.mdl.add_dataset_tensor(input_dataset, acl.MDL_INPUT, data_item)

        # 设置输出数据集
        output_dataset = acl.mdl.create_dataset()
        for buffer in self.output_buffers:
            data_item = acl.create_data_buffer(buffer)
            acl.mdl.add_dataset_tensor(output_dataset, acl.MDL_OUTPUT, data_item)

        # 执行推理
        # 同步执行: 函数会等待推理完成才返回
        ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
        if ret != acl.ACL_SUCCESS:
            raise RuntimeError(f"推理执行失败: {ret}")

        print("✅ 推理执行完成")

        # 清理数据集 (保留缓冲区供下次使用)
        acl.mdl.destroy_dataset(input_dataset)
        acl.mdl.destroy_dataset(output_dataset)

    def get_inference_result(self):
        """
        步骤6: 从设备内存复制推理结果到主机

        Returns:
            numpy.ndarray: 推理输出结果
        """
        output_data = []
        output_num = acl.mdl.get_num_outputs(self.model_desc)

        for i in range(output_num):
            # 从设备复制到主机
            host_buffer = np.empty(self.output_sizes[i], dtype=np.uint8)
            ret = acl.rt.memcpy(
                host_buffer,
                self.output_sizes[i],
                self.output_buffers[i],
                self.output_sizes[i],
                acl.MEMCPY_DEVICE_TO_HOST  # 复制方向: 设备到主机
            )
            if ret != acl.ACL_SUCCESS:
                raise RuntimeError(f"复制输出 {i} 失败: {ret}")

            # 根据实际输出形状重构数组
            # 这里需要根据模型实际输出形状调整
            # ResNet50 通常输出 (1, 1000)
            output_data.append(host_buffer)

        return output_data

    def predict(self, image_path):
        """
        完整的推理流程

        Args:
            image_path: 输入图像路径

        Returns:
            推理结果
        """
        print(f"\n🎯 开始推理: {image_path}")

        # 1. 预处理图像
        image = self.preprocess_image(image_path)
        print(f"✅ 图像预处理完成: {image.shape}")

        # 2. 复制输入到设备
        self.copy_input_to_device(image)
        print("✅ 输入数据已传输到 NPU")

        # 3. 执行推理
        self.execute_inference()
        print("✅ 推理完成")

        # 4. 获取结果
        result = self.get_inference_result()
        print("✅ 结果已获取")

        return result

    def cleanup(self):
        """
        步骤7: 清理资源
        释放所有分配的资源，避免内存泄漏
        """
        print("\n🧹 清理 ACL 资源...")

        # 释放输入缓冲区
        for buffer in self.input_buffers:
            acl.rt.free(buffer)

        # 释放输出缓冲区
        for buffer in self.output_buffers:
            acl.rt.free(buffer)

        # 销毁模型
        if self.model_id is not None:
            acl.mdl.destroy_model(self.model_id)

        # 销毁模型描述符
        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)

        # 销毁流和上下文
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)

        if self.context is not None:
            acl.rt.destroy_context(self.context)

        # 重置设备
        acl.rt.reset_device(self.device_id)

        print("✅ 资源清理完成")


def main():
    """主函数 - 演示完整使用流程"""
    # 模型文件路径 (需要先用 ATC 工具将 ResNet50 转换为 .om 格式)
    model_path = "resnet50.om"

    # 创建推理器实例
    inference = ResNet50ACLInference(model_path=model_path, device_id=0)

    try:
        # 初始化 ACL 环境
        inference.init_acl()

        # 加载模型
        inference.load_model()

        # 分配内存缓冲区
        inference.allocate_buffers()

        # 执行推理
        # 需要有一张 224x224 的测试图像
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            results = inference.predict(test_image)
            print(f"\n📊 推理结果: {len(results)} 个输出")
            print(f"第一个输出大小: {len(results[0])} bytes")
        else:
            print(f"⚠️  测试图像不存在: {test_image}")
            print("请将测试图像命名为 'test_image.jpg' 并放入当前目录")

    except Exception as e:
        print(f"❌ 错误: {e}")

    finally:
        # 确保资源被清理
        inference.cleanup()


if __name__ == "__main__":
    main()