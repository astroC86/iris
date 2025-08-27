#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
import numpy as np
import iris
from iris._mpi_helpers import mpi_allgather
# from examples.common.utils import read_realtime

@triton.jit
def read_realtime():
    tmp = tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints=("=l"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp

@triton.jit()
def gather_latencies(
    local_latency,
    global_latency,
    curr_rank,
    num_ranks ,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    latency_mask = offsets < num_ranks
    iris.put(local_latency + offsets, global_latency +  curr_rank * num_ranks + offsets, curr_rank, 0, heap_bases, mask=latency_mask)

@triton.jit()
def ping_pong(
    data,
    n_elements,
    skip,
    niter,
    flag,
    curr_rank,
    peer_rank,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    data_mask = offsets < n_elements
    flag_mask = offsets < 1
    time_stmp_mask = offsets < 1

    for i in range(niter + skip):
        if i == skip:
            start = read_realtime()
            tl.atomic_xchg(mm_begin_timestamp_ptr + peer_rank * BLOCK_SIZE + offsets, start, time_stmp_mask)
        first_rank = tl.minimum(curr_rank, peer_rank) if (i % 2) == 0 else tl.maximum(curr_rank, peer_rank)
        token_first_done  = i + 1
        token_second_done = i + 2
        if curr_rank == first_rank:
            iris.put(data + offsets, data + offsets, curr_rank, peer_rank, heap_bases, mask=data_mask)
            iris.store(flag + offsets, token_first_done, curr_rank, peer_rank, heap_bases, flag_mask)
            while tl.load(flag, cache_modifier=".cv", volatile=True) != token_second_done:
                pass
        else:
            while tl.load(flag, cache_modifier=".cv", volatile=True) != token_first_done:
                pass
            iris.put(data + offsets, data + offsets, curr_rank, peer_rank, heap_bases, mask=data_mask)
            iris.store(flag + offsets, token_second_done, curr_rank, peer_rank, heap_bases, flag_mask)

    stop = read_realtime()
    tl.atomic_xchg(mm_end_timestamp_ptr + peer_rank * BLOCK_SIZE + offsets, stop, time_stmp_mask)

if __name__ == "__main__":
    dtype     = torch.int32
    heap_size = 1 << 32
    shmem = iris.iris(heap_size)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = 1
    BUFFER_LEN = 1

    iter = 200
    skip = 1
    mm_begin_timestamp = torch.zeros((num_ranks, BLOCK_SIZE), dtype=torch.int64, device="cuda")
    mm_end_timestamp   = torch.zeros((num_ranks, BLOCK_SIZE), dtype=torch.int64, device="cuda")

    local_latency      = torch.zeros((num_ranks), dtype=torch.float32, device="cuda")

    source_buffer = shmem.ones(BUFFER_LEN, dtype=dtype)
    result_buffer = shmem.zeros_like(source_buffer)
    flag          = shmem.ones(1, dtype=dtype)

    grid = lambda meta: (1,)
    for source_rank in range(num_ranks):
        for destination_rank in range(num_ranks):
            if source_rank != destination_rank and cur_rank in [source_rank, destination_rank]:
                peer_for_me = destination_rank if cur_rank == source_rank else source_rank
                ping_pong[grid](source_buffer, 
                                BUFFER_LEN, 
                                skip, iter, 
                                flag, 
                                cur_rank,  peer_for_me,
                                BLOCK_SIZE, 
                                heap_bases, 
                                mm_begin_timestamp, 
                                mm_end_timestamp)
            shmem.barrier()
    
    for destination_rank in range(num_ranks):
        local_latency[destination_rank] = (mm_end_timestamp.cpu()[destination_rank] - mm_begin_timestamp.cpu()[destination_rank]) / iter
    
    latency_matrix = mpi_allgather(local_latency.cpu())

    if cur_rank == 0:
        with open(f"latency.txt", "w") as f:
            f.write(" ," + ", ".join(f"R{j}" for j in range(num_ranks)) + "\n")
            for i in range(num_ranks):
                row_entries = []
                for j in range(num_ranks):
                    val = float(latency_matrix[i, j])
                    row_entries.append(f"{val:0.6f}")
                line = f"R{i}," + ", ".join(row_entries) + "\n"
                f.write(line)