# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tqdm.auto import tqdm
import torch
import torch.distributed as dist

from example.model import GPTConfig, GPT2Model
from tiny_megatron.core import ParallelContext, apply_pipeline_parallel

# init distributed
rank = int(os.getenv('LOCAL_RANK', '0'))
torch.manual_seed(rank)
world_size = int(os.getenv('WORLD_SIZE', '1'))
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(rank)

config = GPTConfig()
model = GPT2Model(config)   # init model on CPU
parallel_context = ParallelContext({"pp": world_size})
model = apply_pipeline_parallel(    # apply PP to transformer blocks
    model=model,
    parallel_context=parallel_context,
    block_names=["transformer.h"]
)
input = torch.randint(0, config.vocab_size, (1, config.block_size)).cuda()
target = torch.randint(0, config.vocab_size, (1, config.block_size)).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-1)

for i in tqdm(range(100)):
    optimizer.zero_grad()
    logits, loss = model(input, target)
    
    # Only last rank will have valid loss, others will have None
    if rank == world_size - 1 and loss is not None:
        tqdm.write(f"iter {i} loss: {loss.item():.4f}")
        loss.backward()
    
    # All ranks participate in optimizer step
    optimizer.step()
    
dist.destroy_process_group() 