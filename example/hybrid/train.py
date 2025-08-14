# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

"""
2D Parallelism Training Example: TP (inner) + DP (outer)

This example demonstrates how to use HybridParallelWrapper for 2D parallelism.
For 8 GPUs: TP=2 (tensor parallel) x DP=4 (data parallel)
For 4 GPUs: TP=2 (tensor parallel) x DP=2 (data parallel)

Usage:
    torchrun --nproc_per_node=8 example/hybrid/train.py
    torchrun --nproc_per_node=4 example/hybrid/train.py
    torchrun --nproc_per_node=2 example/hybrid/train.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tqdm.auto import tqdm
import torch
import torch.distributed as dist

from example.model import GPTConfig, GPT2Model
from tiny_megatron.core import ParallelContext, apply_hybrid_parallel

# init distributed
rank = int(os.getenv('LOCAL_RANK', '0'))
torch.manual_seed(rank)
world_size = int(os.getenv('WORLD_SIZE', '1'))
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(rank)

# configure 2D parallelism based on world size
if world_size == 8:
    parallel_config = {"tp": 2, "dp": 4}  # 2D: 2x4
elif world_size == 4:
    parallel_config = {"tp": 2, "dp": 2}  # 2D: 2x2
elif world_size == 2:
    parallel_config = {"tp": 2, "dp": 1}  # 1D: 2 (TP only)
else:
    raise ValueError(f"Unsupported world_size: {world_size}. Supported: 2, 4, 8")

config = GPTConfig()
model = GPT2Model(config)   # init model on CPU
parallel_context = ParallelContext(parallel_config)

# Apply 2D hybrid parallelism
tp_config = {
    "column_linear_patterns": ["attn.c_attn", "mlp.c_fc"],  # QKV and FC projections
    "row_linear_patterns": ["attn.c_proj", "mlp.c_proj"]   # Output projections
}

model = apply_hybrid_parallel(
    model=model,
    parallel_context=parallel_context,
    tp_config=tp_config
)

input = torch.randint(0, config.vocab_size, (1, config.block_size)).cuda()
target = torch.randint(0, config.vocab_size, (1, config.block_size)).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-1)

for i in tqdm(range(100)):
    optimizer.zero_grad()
    
    # Enable DP gradient sync for this iteration
    if hasattr(model, 'require_backward_grad_sync'):
        model.require_backward_grad_sync = True
    
    logits, loss = model(input, target)
    
    # Loss is always valid for 2D parallelism (no PP)
    if loss is not None:
        tqdm.write(f"iter {i} loss: {loss.item():.4f}")
        loss.backward()
    
    optimizer.step()

dist.destroy_process_group() 