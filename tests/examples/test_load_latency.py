#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
import numpy as np
import iris
from examples.common.utils import read_realtime


@triton.jit()
def ping_pong(
    data,
    result,
    len,
    iter,
    skip,
    flag: tl.tensor,
    curr_rank,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    peer = (curr_rank + 1) % 2
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    data_mask = offsets < len
    flag_mask = offsets < 1
    time_stmp_mask = offsets < 1

    for i in range(iter + skip):
        if i == skip:
            start = read_realtime()
            tl.atomic_xchg(mm_begin_timestamp_ptr + offsets, start, time_stmp_mask)
        if curr_rank == (i + 1) % 2:
            while tl.load(flag, cache_modifier=".cv", volatile=True) != i + 1:
                pass
            iris.put(data + offsets, result + offsets, curr_rank, peer, heap_bases, mask=data_mask)
            tl.store(flag + offsets, i + 1, mask=flag_mask)
            iris.put(flag + offsets, flag + offsets, curr_rank, peer, heap_bases, flag_mask)
        else:
            iris.put(data + offsets, result + offsets, curr_rank, peer, heap_bases, mask=data_mask)
            tl.store(flag + offsets, i + 1, mask=flag_mask)
            iris.put(flag + offsets, flag + offsets, curr_rank, peer, heap_bases, flag_mask)
            while tl.load(flag, cache_modifier=".cv", volatile=True) != i + 1:
                pass
    stop = read_realtime()
    tl.atomic_xchg(mm_end_timestamp_ptr + offsets, stop, time_stmp_mask)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        # torch.float16,
        # torch.bfloat16,
        # torch.float32,
    ],
)
@pytest.mark.parametrize(
    "heap_size",
    [
        (1 << 33),
    ],
)
def test_load_bench(dtype, heap_size):
    shmem = iris.iris(heap_size)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()
    assert num_ranks == 2

    BLOCK_SIZE = 1
    BUFFER_LEN = 64 * 1024

    iter = 200
    skip = 20
    mm_begin_timestamp = torch.zeros(BLOCK_SIZE, dtype=torch.int64, device="cuda")
    mm_end_timestamp = torch.zeros(BLOCK_SIZE, dtype=torch.int64, device="cuda")

    source_buffer = shmem.ones(BUFFER_LEN, dtype=dtype)
    result_buffer = shmem.zeros_like(source_buffer)
    flag = shmem.ones(1, dtype=dtype)

    grid = lambda meta: (1,)
    ping_pong[grid](
        source_buffer,
        result_buffer,
        BUFFER_LEN,
        skip,
        iter,
        flag,
        cur_rank,
        BLOCK_SIZE,
        heap_bases,
        mm_begin_timestamp,
        mm_end_timestamp,
    )
    shmem.barrier()
    begin_val = mm_begin_timestamp.cpu().item()
    end_val = mm_end_timestamp.cpu().item()
    with open(f"timestamps_{cur_rank}.txt", "w") as f:
        f.write(f"mm_begin_timestamp: {begin_val}\n")
        f.write(f"mm_end_timestamp: {end_val}\n")
