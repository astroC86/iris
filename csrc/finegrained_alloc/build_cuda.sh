#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

name="finegrained_allocator"

# Warnings forwarded to host compiler (GCC/Clang)
basic_warnings="-Xcompiler=-Wall -Xcompiler=-Wextra"

strict_warnings="-Xcompiler=-Wshadow \
 -Xcompiler=-Wnon-virtual-dtor \
 -Xcompiler=-Wold-style-cast \
 -Xcompiler=-Wcast-align \
 -Xcompiler=-Woverloaded-virtual \
 -Xcompiler=-Wconversion \
 -Xcompiler=-Wsign-conversion \
 -Xcompiler=-Wnull-dereference \
 -Xcompiler=-Wdouble-promotion \
 -Xcompiler=-Wformat=2"

# NVCC supports -std=c++17 directly
std_flags="-std=c++17"

# Output settings
output_flags="-Xcompiler=-fPIC -shared -o lib${name}.so"

nvcc -arch=sm_90 $basic_warnings $strict_warnings $std_flags $output_flags ${name}.cu