# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
import torch.distributed as dist

from ...module import (
    ops,
    linear, 
)
from .utils import Parameter, divide
from ..utils.comm import ParallelContext

# inspired from https://arxiv.org/pdf/1909.08053

def sync_tensor(tensor, async_op=True, group=None):    # communication complexity: 2g
    if async_op:
        work = dist.all_reduce(tensor, async_op=True, group=group)
    else:
        dist.all_reduce(tensor, async_op=False, group=group)
        work = None
    return work


def shard_tensor(tensor, axis: int, num_shards: int, rank: int):
    assert tensor.size(axis) % num_shards == 0
    per_part = tensor.size(axis) // num_shards
    start = rank * per_part
    end = start + per_part
    return torch.narrow(tensor, dim=axis, start=start, length=per_part)

def gather_tensor(tensor, axis, group, world_size, rank):
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=axis)



class ColumnParallelLinear(linear.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, 
                 parallel_context: ParallelContext = None, auto_tune=False):
        if parallel_context is None:
            raise ValueError("parallel_context must be provided for ColumnParallelLinear")
            
        self.context = parallel_context
        self.tp_size = self.context.parallel_dims["tp"]
        self.tp_rank = self.context.get_rank_in("tp")
        self.tp_group = self.context.get_group("tp")

        # Shard output dim (axis=0)
        local_out = out_features // self.tp_size
        super().__init__(in_features, local_out, bias, device, dtype, auto_tune)

    def forward_callback(self, ctx, input, weight, bias, runtime_tuner):
        ctx.save_for_backward(input, weight, bias)
        output = linear.linear_forward(input, weight, bias, runtime_tuner)
        return ctx, output  # no communication in fwd

    def backward_callback(self, ctx, grad_output, runtime_tuner):
        input, weight, bias = ctx.saved_tensors

        grad_input = linear.linear_input_grad(grad_output, input, weight, runtime_tuner) \
            if ctx.needs_input_grad[0] else None
        grad_weight = linear.linear_weight_grad(grad_output, input, weight, runtime_tuner) \
            if ctx.needs_input_grad[1] else None
        grad_bias = linear.linear_bias_grad(grad_output, input, weight, runtime_tuner) \
            if bias is not None and ctx.needs_input_grad[2] else None

        # Apply All-Reduce to grad_input if needed (g in the figure)
        if grad_input is not None:
            dist.all_reduce(grad_input, group=self.tp_group)

        # Check if the grad shape is correct
        if grad_input is not None and grad_input.shape != input.shape:
            raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
        if grad_bias is not None and grad_bias.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")
        
        return grad_input, grad_weight, grad_bias


class RowParallelLinear(linear.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, 
                 parallel_context: ParallelContext = None, auto_tune=False):
        if parallel_context is None:
            raise ValueError("parallel_context must be provided for RowParallelLinear")
            
        self.context = parallel_context
        self.tp_size = self.context.parallel_dims["tp"]
        self.tp_rank = self.context.get_rank_in("tp")
        self.tp_group = self.context.get_group("tp")

        # Shard input dim (axis=1)
        local_in = in_features // self.tp_size
        super().__init__(local_in, out_features, bias, device, dtype, auto_tune)

    def forward_callback(self, ctx, input, weight, bias, runtime_tuner):
        # Save original input shape for backward
        original_input = input
        
        # f: all-gather input from all TP ranks
        gathered = [torch.zeros_like(input) for _ in range(self.tp_size)]
        dist.all_gather(gathered, input, group=self.tp_group)
        input_gathered = torch.cat(gathered, dim=-1)
        
        # Local matrix multiplication
        output = linear.linear_forward(input_gathered, weight, bias, runtime_tuner)
        
        # f: all-reduce output across TP ranks
        dist.all_reduce(output, group=self.tp_group)
        
        ctx.save_for_backward(original_input, input_gathered, weight, bias)
        return ctx, output

    def backward_callback(self, ctx, grad_output, runtime_tuner):
        original_input, input_gathered, weight, bias = ctx.saved_tensors

        grad_input_full = linear.linear_input_grad(grad_output, input_gathered, weight, runtime_tuner) \
            if ctx.needs_input_grad[0] else None
        grad_weight = linear.linear_weight_grad(grad_output, input_gathered, weight, runtime_tuner) \
            if ctx.needs_input_grad[1] else None
        grad_bias = linear.linear_bias_grad(grad_output, input_gathered, weight, runtime_tuner) \
            if bias is not None and ctx.needs_input_grad[2] else None

        # f: slice grad_input back to local input size
        if grad_input_full is not None:
            chunk_size = grad_input_full.shape[-1] // self.tp_size
            start = self.tp_rank * chunk_size
            end = start + chunk_size
            grad_input = grad_input_full[..., start:end]
        else:
            grad_input = None

        # Check if the grad shape is correct
        if grad_input is not None and grad_input.shape != original_input.shape:
            raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {original_input.shape}")
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
        if grad_bias is not None and grad_bias.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")
        
        return grad_input, grad_weight, grad_bias