#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
op tiling interface
"""

import math
import os
import ctypes
import json
import struct
import hashlib
import threading
import platform
from pathlib import Path

import tbe.common.context.op_context as op_context
from tbe.common.utils.errormgr import get_error_message
from tbe.common.context.op_context import OpContext
from tbe.common.context.op_info import OpInfo
from tbe.common.utils import log

_RT_BANK_CACHE = ""
_MAX_RUN_INFO_SIZE = 1024*64
_CONST_VALUE = "const_value"
_CONST_VALUE_NULL_DESC = "const_value_null_desc"
_ATTR_DTYPE = "dtype"
_ATTR_VALUE = "value"
_ATTR_VALUE_NULL_DESC = "value_null_desc"
_ASCEND_OPP_PATH_ENV = "ASCEND_OPP_PATH"
_ASCEND_OPP_PATH_DEFAULT = os.path.abspath("/usr/local/Ascend/latest/opp")
_ASCEND_OPP_PATH_DEFAULT_MDC = os.path.abspath("/usr/local/Ascend/opp")

# Tiling is likely running in thread pool or single-threaded process,
# using thread local buffer reduces memory allocation
_TILING_DATA = threading.local()

# Initializing thread local data when importing this py module,
# which is helpful in case of single-threaded profiling test
_TILING_DATA.buf = ctypes.create_string_buffer(_MAX_RUN_INFO_SIZE)
_TILING_DATA.buf_size = ctypes.c_size_t(_MAX_RUN_INFO_SIZE)

platform_arch = platform.machine()
opp_dir = os.environ.get(_ASCEND_OPP_PATH_ENV, _ASCEND_OPP_PATH_DEFAULT)
scene_info = os.path.join(opp_dir, "scene.info")
# mdc default path
scene_info_mdc = os.path.join(_ASCEND_OPP_PATH_DEFAULT_MDC, "scene.info")
# all in one default path
scene_info_path_default = os.path.join(_ASCEND_OPP_PATH_DEFAULT, "scene.info")
# first choose all in one default path
scene_info_default_path = scene_info_path_default if os.path.exists(scene_info_path_default) else scene_info_mdc
# first use opp path env path
scene_info_path = scene_info if os.path.exists(scene_info) else scene_info_default_path
conf_dir = os.path.join(opp_dir, "vendors")
config = os.path.join(opp_dir, "vendors", "config.ini")
op_impl_path = os.path.join("built-in", "op_impl") if os.path.exists(conf_dir) else os.path.join("op_impl", "built-in")
tiling_full_path = os.path.join(opp_dir, op_impl_path, "ai_core", "tbe", "op_tiling", "liboptiling.so")
tiling_so_path = os.path.join(op_impl_path, "ai_core", "tbe", "op_tiling", "liboptiling.so")
tiling_cust_path = os.path.join("ai_core", "tbe", "op_tiling", "liboptiling.so")
#Get system info
# Initialize sys_version with default value to fix CANN 7.1.0 bug
sys_version = "linux"  # Default OS version for Ascend platform
if os.path.exists(scene_info_path):
    with open(scene_info_path) as f:
        scene_info = list(map(lambda x: x.strip(), f.readlines()))
        os_state = False
        for item_info in scene_info:
            if "os=" in item_info:
                sys_version = item_info.split("=")[-1]
                os_state = True
        if os_state is False:
            raise RuntimeError({"errCode": "E80001", "config_name":"os", "file_name": "scene.info"})
tiling_so_arch_path = os.path.join("ai_core", "tbe", "op_tiling", "lib", sys_version, platform_arch, "liboptiling.so")
tiling_so_arch_path2 =\
    os.path.join("ai_core", "tbe", "op_tiling", "lib", sys_version, platform_arch, "libopmaster_rt2.0.so")
so_arch_path = os.path.join(op_impl_path, tiling_so_arch_path)
so_arch_path2 = os.path.join(op_impl_path, tiling_so_arch_path2)
_BUILTIN_TILING_PATH = tiling_so_path if os.path.exists(tiling_full_path) else so_arch_path

if os.path.exists(config):
    with open(config) as f:
        _VENDOR_NAME = f.readline().split("=")[1].split(",")[0].strip()
        _CUSTOM_TILING_PATH_DEFAULT = os.path.join("vendors", _VENDOR_NAME, "op_impl", tiling_cust_path)
else:
    _VENDOR_NAME = "customize"
    _CUSTOM_TILING_PATH_DEFAULT = os.path.join("op_impl", "custom", tiling_cust_path)


def _gen_null_desc(value_list):
    if not isinstance(value_list, list):
        return None
    value_null_desc = []
    is_exist_null = False
    for idx, value in enumerate(value_list):
        if not isinstance(value, float):
            continue
        if value == float("inf"):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("inf")
        elif value == float("-inf"):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("-inf")
        elif math.isnan(value):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("nan")
        else:
            value_null_desc.append(None)

    return value_null_desc if is_exist_null else None


def _inputs_pre_process(inputs):
    if not isinstance(inputs, (list, tuple)):
        return
    for single_input in inputs:
        if not isinstance(single_input, dict):
            continue
        const_value = single_input.get(_CONST_VALUE)
        if not isinstance(const_value, (list, tuple)):
            continue
        const_value_list = list(const_value)
        const_value_null_desc = _gen_null_desc(const_value_list)
        if const_value_null_desc is not None:
            single_input[_CONST_VALUE] = const_value_list
            single_input[_CONST_VALUE_NULL_DESC] = const_value_null_desc


def _attrs_pre_process(attrs):
    if not isinstance(attrs, (list, tuple)):
        return
    for single_attr in attrs:
        if not isinstance(single_attr, dict):
            continue
        attr_dtype = single_attr.get(_ATTR_DTYPE)
        if attr_dtype not in ("float", "float32", "list_float", "list_float32"):
            continue
        attr_value = single_attr.get(_ATTR_VALUE)
        if attr_value is None:
            continue
        is_single_element = False
        if not isinstance(attr_value, (list, tuple)):
            is_single_element = True
            attr_value = [attr_value]
        attr_value_list = list(attr_value)
        attr_null_desc = _gen_null_desc(attr_value_list)
        if attr_null_desc is not None:
            if is_single_element:
                single_attr[_ATTR_VALUE_NULL_DESC] = attr_null_desc[0]
                single_attr[_ATTR_VALUE] = attr_value_list[0]
            else:
                single_attr[_ATTR_VALUE_NULL_DESC] = attr_null_desc
                single_attr[_ATTR_VALUE] = attr_value_list


def do_op_tiling(optype, compile_info, inputs, outputs, compile_info_hash=None, timer=None, attrs=None):
    """
    do op tilinng
    """
    def _load_lib():
        opp_path = Path(os.environ.get(_ASCEND_OPP_PATH_ENV, _ASCEND_OPP_PATH_DEFAULT))
        builtin_optiling_lib_path = opp_path.joinpath(_BUILTIN_TILING_PATH)
        builtin_optiling_lib_path2 = opp_path.joinpath(so_arch_path2)
        custom_optiling_lib_path = opp_path.joinpath(_CUSTOM_TILING_PATH_DEFAULT)

        libregister = ctypes.CDLL("libregister.so")

        # 1. builint optiling 1.0 regist
        if os.path.exists(builtin_optiling_lib_path):
            ctypes.CDLL(builtin_optiling_lib_path)

        # 2. custom optiling 2.0 regist | custom optiling 1.0 regist
        try:
            lib_optiling = ctypes.CDLL(custom_optiling_lib_path)
            custom_opp_so_path_str = str(custom_optiling_lib_path)
            lib_optiling.TbeLoadSoAndSaveToRegistry(custom_opp_so_path_str.encode('utf_8'))
        except OSError:
            # Custom op tiling lib may not exists
            pass

        # 3. builtin optiling 2.0 regist
        if os.path.exists(builtin_optiling_lib_path2):
            nonlocal lib_optiling_builtin
            lib_optiling_builtin = ctypes.CDLL(builtin_optiling_lib_path2)
            builtin_optiling_lib_path2_str = str(builtin_optiling_lib_path2)
            lib_optiling_builtin.TbeLoadSoAndSaveToRegistry(builtin_optiling_lib_path2_str.encode('utf_8'))

        return libregister

    if isinstance(op_context.get_context(), OpContext) and \
       isinstance(op_context.get_context().get_graph_op_info(), OpInfo) and \
        (op_context.get_context().get_graph_op_info().op_name is not None):
        op_name = op_context.get_context().get_graph_op_info().op_name
    else:
        op_name = ""
    extra_params = {"op_name": op_name}
    lib_optiling_builtin = None
    libregister = _load_lib()
    if _RT_BANK_CACHE:
        pid = os.getpid()
        pid_c = ctypes.c_uint32(pid)
        optype_c = optype.encode('utf_8')
        tiling_c = _RT_BANK_CACHE.encode('utf_8')
        log.info(f"Start to do SetTuningTiling of {optype}, tiling: {_RT_BANK_CACHE}.")
        set_tiling_func = lib_optiling_builtin.SetTuningTiling
        if set_tiling_func(pid_c, optype_c, tiling_c) != 0:
            log.error(f"SetTuningTiling of {optype} failed, tiling: {_RT_BANK_CACHE}.")

    _inputs_pre_process(inputs)
    _attrs_pre_process(attrs)
    optype_c = optype.encode('utf_8')
    compile_info_c = json.dumps(compile_info).encode('utf_8')
    inputs_c = json.dumps(inputs).encode('utf_8')
    outputs_c = json.dumps(outputs).encode('utf_8')
    extra_params_c = json.dumps(extra_params).encode('utf_8')
    # Attrs supported format: ({name: str, dtype: str, value: Any}, ...)
    # Attrs supported dtypes: (bool, float, float32, int, int32, list_bool, list_float, list_float32, list_int,
    #                          list_int32, list_list_int, list_list_int32, list_str, str)
    if not attrs:
        attrs_c = ctypes.c_void_p()
    else:
        attrs_c = json.dumps(attrs).encode('utf_8')
    if not compile_info_hash:
        hashstr = hashlib.sha1()
        hashstr.update(compile_info_c)
        compile_info_hash = hashstr.hexdigest()
    compile_info_hash_c = compile_info_hash.encode('utf_8')

    if not hasattr(_TILING_DATA, "buf") or not hasattr(_TILING_DATA, "buf_size"):
        _TILING_DATA.buf = ctypes.create_string_buffer(_MAX_RUN_INFO_SIZE)
        _TILING_DATA.buf_size = ctypes.c_size_t(_MAX_RUN_INFO_SIZE)

    tiling_func = libregister.OpTilingForCompile
    if isinstance(timer, list):
        array_c = ctypes.c_uint64 * 3
        elapse_c = array_c(0, 0, 0)
        res = tiling_func(optype_c, compile_info_c, compile_info_hash_c,
                          inputs_c, outputs_c, attrs_c,
                          _TILING_DATA.buf, _TILING_DATA.buf_size,
                          elapse_c, extra_params_c)
        for i in range(0, 3):
            timer.append(elapse_c[i])
    else:
        res = tiling_func(optype_c, compile_info_c, compile_info_hash_c,
                          inputs_c, outputs_c, attrs_c,
                          _TILING_DATA.buf, _TILING_DATA.buf_size,
                          ctypes.c_void_p(), extra_params_c)
    if not res:
        dict_args = {}
        inputs_str = "\n".join(tuple(map(str, inputs)))
        outputs_str = "\n".join(tuple(map(str, outputs)))
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = f"Tiling func of op_type {optype} failed, failure details:\n" \
                                      f"Compile_info: {compile_info}\n" \
                                      f"Inputs: {inputs_str}\n" \
                                      f"Outputs: {outputs_str}\n" \
                                      f"[OP_TILING] Attrs: {attrs}"
        raise RuntimeError(dict_args, get_error_message(dict_args))

    run_info = json.loads(_TILING_DATA.buf.value)
    run_info['tiling_data'] = bytes.fromhex(run_info['tiling_data'])
    return run_info


def decode(tiling_data, fmt):
    """
    decode tiling data
    """
    offset = 0

    def _get_value(tiling_data, fmt, offset=0):
        """
        fmt example: [-1, "int"]   # int arrary of unknown size
                     [10, "int"]   # arrary of 10 ints
                     "int"         # single int
        """
        fmt_def = {
            "char": "c",
            "int": "i",
            "int32": "i",
            "uint": "I",
            "int8": "b",
            "uint8": "B",
            "int16": "h",
            "uint16": "H",
            "int64": "l",
            "uint64": "L",
            "double": "d"
        }
        count = 1
        unpack_size = 0
        if isinstance(fmt, (list, tuple)):
            count = fmt[0]
            if count < 0:
                fmt_size = struct.calcsize("i")
                res = struct.unpack_from("i", tiling_data, offset)
                count = res[0]
                unpack_size += fmt_size
            fmt_str = "{}{}".format(count, fmt_def.get(fmt[1]))
        else:
            fmt_str = "{}{}".format(count, fmt_def.get(fmt))

        if count == 0:
            return [unpack_size, []]

        fmt_size = struct.calcsize(fmt_str)
        res = struct.unpack_from(fmt_str, tiling_data, offset + unpack_size)
        unpack_size += fmt_size
        if isinstance(fmt, (list, tuple)):
            return [unpack_size, res]
        return [unpack_size, res[0]]

    res = {}
    for key, value in fmt.items():
        unpack_size, res[key] = _get_value(tiling_data, value, offset)
        offset += unpack_size

    return res
