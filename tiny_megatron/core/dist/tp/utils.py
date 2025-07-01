# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
import torch.distributed as dist

class Parameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, bwd_sync=False, rank_id=None):
        t = nn.Parameter.__new__(cls, data, requires_grad)
        t.bwd_sync = bwd_sync
        t.rank_id = rank_id
        return t

# modified from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/utils.py

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
