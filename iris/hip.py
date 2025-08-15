# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import enum
import numpy as np
import sys
import torch

rt_path = "libamdhip64.so"
hip_runtime = ctypes.cdll.LoadLibrary(rt_path)

amdsmi_path = 'libamd_smi.so'
amdsmi = ctypes.cdll.LoadLibrary(amdsmi_path)


class amdsmi_status_t(enum.IntEnum):
    AMDSMI_STATUS_SUCCESS = 0x0
    AMDSMI_STATUS_INVALID_ARGS = 0x1
    AMDSMI_STATUS_NOT_SUPPORTED = 0x2
    AMDSMI_STATUS_FILE_ERROR = 0x3
    AMDSMI_STATUS_PERMISSION = 0x4
    AMDSMI_STATUS_OUT_OF_RESOURCES = 0x5
    AMDSMI_STATUS_INTERNAL_EXCEPTION = 0x6
    AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS = 0x7
    AMDSMI_STATUS_INIT_ERROR = 0x8
    AMDSMI_INITIALIZATION_ERROR = AMDSMI_STATUS_INIT_ERROR
    AMDSMI_STATUS_NOT_YET_IMPLEMENTED = 0x9
    AMDSMI_STATUS_NOT_FOUND = 0xA
    AMDSMI_STATUS_INSUFFICIENT_SIZE = 0xB
    AMDSMI_STATUS_INTERRUPT = 0xC
    AMDSMI_STATUS_UNEXPECTED_SIZE = 0xD
    AMDSMI_STATUS_NO_DATA = 0xE
    AMDSMI_STATUS_UNEXPECTED_DATA = 0xF
    AMDSMI_STATUS_BUSY = 0x10
    AMDSMI_STATUS_REFCOUNT_OVERFLOW = 0x11
    AMDSMI_STATUS_SETTING_UNAVAILABLE = 0x12
    AMDSMI_STATUS_AMDGPU_RESTART_ERR = 0x13
    AMDSMI_STATUS_DRM_ERROR = 0x14
    AMDSMI_STATUS_FAIL_LOAD_MODULE = 0x15
    AMDSMI_STATUS_FAIL_LOAD_SYMBOL = 0x16
    AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF


def hip_try(err):
    if err != 0:
        hip_runtime.hipGetErrorString.restype = ctypes.c_char_p
        error_string = hip_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
        raise RuntimeError(f"HIP error code {err}: {error_string}")


def _amdsmi_try(err):
    if err != amdsmi_status_t.AMDSMI_STATUS_SUCCESS:
        error_messages = {
            amdsmi_status_t.AMDSMI_STATUS_INVALID_ARGS: "AMDSMI_STATUS_INVALID_ARGS - Invalid parameters",
            amdsmi_status_t.AMDSMI_STATUS_NOT_SUPPORTED: "AMDSMI_STATUS_NOT_SUPPORTED - Command not supported",
            amdsmi_status_t.AMDSMI_STATUS_FILE_ERROR: "AMDSMI_STATUS_FILE_ERROR - Problem accessing a file",
            amdsmi_status_t.AMDSMI_STATUS_PERMISSION: "AMDSMI_STATUS_PERMISSION - Permission Denied",
            amdsmi_status_t.AMDSMI_STATUS_OUT_OF_RESOURCES: "AMDSMI_STATUS_OUT_OF_RESOURCES - Not enough memory or resources",
            amdsmi_status_t.AMDSMI_STATUS_INTERNAL_EXCEPTION: "AMDSMI_STATUS_INTERNAL_EXCEPTION - An internal exception was caught",
            amdsmi_status_t.AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS: "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS - The provided input is out of allowable or safe range",
            amdsmi_status_t.AMDSMI_STATUS_INIT_ERROR: "AMDSMI_STATUS_INIT_ERROR - An error occurred when initializing internal data structures",
            amdsmi_status_t.AMDSMI_STATUS_NOT_YET_IMPLEMENTED: "AMDSMI_STATUS_NOT_YET_IMPLEMENTED - Not implemented yet",
            amdsmi_status_t.AMDSMI_STATUS_NOT_FOUND: "AMDSMI_STATUS_NOT_FOUND - Device or resource not found",
            amdsmi_status_t.AMDSMI_STATUS_INSUFFICIENT_SIZE: "AMDSMI_STATUS_INSUFFICIENT_SIZE - Not enough resources were available for the operation",
            amdsmi_status_t.AMDSMI_STATUS_INTERRUPT: "AMDSMI_STATUS_INTERRUPT - An interrupt occurred during execution of function",
            amdsmi_status_t.AMDSMI_STATUS_UNEXPECTED_SIZE: "AMDSMI_STATUS_UNEXPECTED_SIZE - An unexpected amount of data was read",
            amdsmi_status_t.AMDSMI_STATUS_NO_DATA: "AMDSMI_STATUS_NO_DATA - No data was found for a given input",
            amdsmi_status_t.AMDSMI_STATUS_UNEXPECTED_DATA: "AMDSMI_STATUS_UNEXPECTED_DATA - The data read or provided to function is not what was expected",
            amdsmi_status_t.AMDSMI_STATUS_BUSY: "AMDSMI_STATUS_BUSY - Device is busy",
            amdsmi_status_t.AMDSMI_STATUS_REFCOUNT_OVERFLOW: "AMDSMI_STATUS_REFCOUNT_OVERFLOW - An internal reference counter exceeded maximum value",
            amdsmi_status_t.AMDSMI_STATUS_SETTING_UNAVAILABLE: "AMDSMI_STATUS_SETTING_UNAVAILABLE - Setting is not available",
            amdsmi_status_t.AMDSMI_STATUS_AMDGPU_RESTART_ERR: "AMDSMI_STATUS_AMDGPU_RESTART_ERR - AMDGPU restart failed",
            amdsmi_status_t.AMDSMI_STATUS_DRM_ERROR: "AMDSMI_STATUS_DRM_ERROR - Error when calling libdrm",
            amdsmi_status_t.AMDSMI_STATUS_FAIL_LOAD_MODULE: "AMDSMI_STATUS_FAIL_LOAD_MODULE - Failed to load library module",
            amdsmi_status_t.AMDSMI_STATUS_FAIL_LOAD_SYMBOL: "AMDSMI_STATUS_FAIL_LOAD_SYMBOL - Failed to load symbol",
            amdsmi_status_t.AMDSMI_STATUS_UNKNOWN_ERROR: "AMDSMI_STATUS_UNKNOWN_ERROR - An unknown error occurred",
        }
        error_string = error_messages.get(err, "AMDSMI_STATUS_UNKNOWN_ERROR - An unknown error occurred")
        raise RuntimeError(f"AMDSMI error code {err}: {error_string}")


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def open_ipc_handle(ipc_handle_data, rank):
    ptr = ctypes.c_void_p()
    hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
    hip_runtime.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        hipIpcMemHandle_t,
        ctypes.c_uint,
    ]
    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 64:
            raise ValueError("ipc_handle_data must be a 64-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * 64).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError("ipc_handle_data must be a numpy.ndarray of dtype uint8 with 64 elements")

    raw_memory = ctypes.create_string_buffer(64)
    ctypes.memset(raw_memory, 0x00, 64)
    ipc_handle_struct = hipIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, 64)

    hip_try(
        hip_runtime.hipIpcOpenMemHandle(
            ctypes.byref(ptr),
            ipc_handle_struct,
            hipIpcMemLazyEnablePeerAccess,
        )
    )

    return ptr.value


def get_ipc_handle(ptr, rank):
    ipc_handle = hipIpcMemHandle_t()
    hip_try(hip_runtime.hipIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    return ipc_handle


def count_devices():
    device_count = ctypes.c_int()
    hip_try(hip_runtime.hipGetDeviceCount(ctypes.byref(device_count)))
    return device_count.value


def set_device(gpu_id):
    hip_try(hip_runtime.hipSetDevice(gpu_id))


def get_device_id():
    device_id = ctypes.c_int()
    hip_try(hip_runtime.hipGetDevice(ctypes.byref(device_id)))
    return device_id.value


def get_cu_count(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    hipDeviceAttributeMultiprocessorCount = 63
    cu_count = ctypes.c_int()

    hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(cu_count), hipDeviceAttributeMultiprocessorCount, device_id))

    return cu_count.value


# Starting ROCm 6.5
# def get_xcc_count(device_id=None):
#     if device_id is None:
#         device_id = get_device()

#     hipDeviceAttributeNumberOfXccs = ??
#     xcc_count = ctypes.c_int()

#     hip_try(hip_runtime.hipDeviceGetAttribute(
#         ctypes.byref(xcc_count),
#         hipDeviceAttributeNumberOfXccs,
#         device_id
#     ))

#     return xcc_count


def get_wall_clock_rate(device_id):
    hipDeviceAttributeWallClockRate = 10017
    wall_clock_rate = ctypes.c_int()
    status = hip_runtime.hipDeviceGetAttribute(
        ctypes.byref(wall_clock_rate), hipDeviceAttributeWallClockRate, device_id
    )
    hip_try(status)
    return wall_clock_rate.value


def get_arch_string(device_id = None):
    if device_id is None:
        device_id = get_device_id()
    arch_full = torch.cuda.get_device_properties(device_id).gcnArchName
    arch_name = arch_full.split(':')[0]
    return arch_name


def _amdsmi_init():
    AMDSMI_INIT_AMD_GPUS = 2
    amdsmi.amdsmi_init.argtypes = [ctypes.c_uint64]
    _amdsmi_try(amdsmi.amdsmi_init(ctypes.c_uint64(AMDSMI_INIT_AMD_GPUS)))


def _amdsmi_shutdown():
    amdsmi.amdsmi_shut_down.argtypes = []
    _amdsmi_try(amdsmi.amdsmi_shut_down())


def get_num_xcd(device_id = None):
    if device_id is None:
        device_id = get_device_id()
    _amdsmi_init()  
    xcd_counter = ctypes.c_uint16(0)
    amdsmi.rsmi_dev_metrics_xcd_counter_get.argtypes = [ ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16)]
    _amdsmi_try(
        amdsmi.rsmi_dev_metrics_xcd_counter_get(
            ctypes.c_uint32(device_id), 
            ctypes.byref(xcd_counter)
        )
    )
    _amdsmi_shutdown()      
    return xcd_counter.value


def malloc_fine_grained(size):
    hipDeviceMallocFinegrained = 0x1
    ptr = ctypes.c_void_p()
    hip_try(hip_runtime.hipExtMallocWithFlags(ctypes.byref(ptr), size, hipDeviceMallocFinegrained))
    return ptr


def hip_malloc(size):
    ptr = ctypes.c_void_p()
    hip_try(hip_runtime.hipMalloc(ctypes.byref(ptr), size))
    return ptr


def hip_free(ptr):
    hip_try(hip_runtime.hipFree(ptr))
