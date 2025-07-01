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

# inspired from https://arxiv.org/pdf/1909.08053

def sync_tensor(tensor, async_op=True, group=None):    # communication complexity: 2g
    if async_op:
        work = dist.all_reduce(tensor, async_op=True, group=group)
    else:
        dist.all_reduce(tensor, async_op=False, group=group)
        work = None
    return work


class ColumnParallelLinear(linear.Linear):
    def _init_parameters(self):
        world_size = dist.get_world_size()
        self.input_size_per_partition = self.in_features
        self.output_size_per_partition = divide(self.out_features, world_size)
        self.allreduce_dx = True
        self.allreduce_out = False
        self.weight = Parameter(torch.empty((self.output_size_per_partition, self.input_size_per_partition), **self.factory_kwargs))
        if self.use_bias:
            self.bias = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
class RowParallelLinear(linear.Linear):
    def _init_parameters(self):
        world_size = dist.get_world_size()
        self.input_size_per_partition = divide(self.in_features, world_size)
        self.output_size_per_partition = self.out_features
        self.allreduce_dx = True
        self.allreduce_out = False
        self.weight = Parameter(torch.empty((self.output_size_per_partition, self.input_size_per_partition), **self.factory_kwargs))
        if self.use_bias:
            self.bias = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
class TensorParallelLinear(linear.Linear):
    def _init_parameters(self):
        self.input_size_per_partition = self.in_features
        self.output_size_per_partition = self.out_features
        self.allreduce_dx = True
        self.allreduce_out = False
        self.weight = Parameter(torch.empty((self.output_size_per_partition, self.input_size_per_partition), **self.factory_kwargs))
        if self.use_bias:
            self.bias = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def _init_comm_group(self, group=None):
        self.group = group
    
    def forward_callback(self, ctx, input, weight, bias, runtime_tuner):
        ctx.save_for_backward(input, weight, bias)
        output = ops.linear_forward(input, weight, bias, runtime_tuner)
        if self.allreduce_out:
            sync_tensor(output, async_op=False, group=self.group)
        return ctx, output

    def backward_callback(self, ctx, grad_output, runtime_tuner):
        input, weight, bias = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = ops.linear_input_grad(grad_output, input, weight, runtime_tuner)
            if self.allreduce_dx:
                handle_dx = sync_tensor(grad_input, group=self.group)
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            grad_weight = ops.linear_weight_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = ops.linear_bias_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_bias = None
        
        # Communication-computation overlap, wait for the communication to finish (core step of tp)
        if ctx.needs_input_grad[0] and self.allreduce_dx and handle_dx is not None:
            handle_dx.wait()
            torch.cuda.synchronize()

        # Check if the grad shape is correct
        if grad_input is not None and grad_input.shape != input.shape:
            raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
        if grad_bias is not None and grad_bias.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")
        
        return grad_input, grad_weight, grad_bias