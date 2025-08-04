# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any

from ..tp.wrapper import TPWrapper
from ..dp.wrapper import DPWrapper
from ..pp.wrapper import PPWrapper
from ..utils.comm import ParallelContext


class HybridParallelWrapper(nn.Module):
    """
    Hybrid Parallel Wrapper supporting 3D parallelism: TP + DP + PP.
    
    This wrapper applies three types of parallelism:
    - Pipeline Parallel (PP): Splits model across pipeline stages
    - Tensor Parallel (TP): Splits individual layers across ranks
    - Data Parallel (DP): Replicates model and synchronizes gradients
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 parallel_context: ParallelContext,
                 tp_config: Optional[Dict[str, Any]] = None,
                 pp_config: Optional[Dict[str, Any]] = None,
                 auto_tune: bool = False):
        """
        Initialize the Hybrid Parallel wrapper.
        
        Args:
            model (nn.Module): The original model to wrap
            parallel_context (ParallelContext): The parallel context containing TP, DP, and PP configs
            tp_config (Dict, optional): TP configuration containing:
                - column_linear_names (List[str]): Names of modules for column parallel
                - row_linear_names (List[str]): Names of modules for row parallel
            pp_config (Dict, optional): PP configuration containing:
                - block_names (List[str]): Names of block containers to distribute across PP ranks
            auto_tune (bool): Whether to enable auto tuning
        """
        super().__init__()
        
        if parallel_context is None:
            raise ValueError("parallel_context must be provided for Hybrid wrapper")
        
        self.parallel_context = parallel_context
        self.auto_tune = auto_tune
        
        # Extract parallel sizes from parallel context
        self.tp_size = parallel_context.parallel_dims.get("tp", 1)
        self.dp_size = parallel_context.parallel_dims.get("dp", 1)
        self.pp_size = parallel_context.parallel_dims.get("pp", 1)
        
        # Validate configuration
        total_parallel_dims = sum([1 for size in [self.tp_size, self.dp_size, self.pp_size] if size > 1])
        if total_parallel_dims == 0:
            raise ValueError("At least one of TP, DP, or PP must have size > 1 for hybrid parallelism")
        
        # Store original model reference
        self.original_model = model
        
        # Apply parallelism layers
        self.model = self._apply_parallelism(model, tp_config, pp_config)
        
        # Track if we need gradient synchronization (for DP)
        self.require_backward_grad_sync = False
        
        print(f"[Rank {parallel_context.rank}] HybridParallelWrapper initialized:")
        print(f"  - TP size: {self.tp_size}, DP size: {self.dp_size}, PP size: {self.pp_size}")
        print(f"  - Coordinates: {parallel_context.get_coord_dict()}")
    
    def _apply_parallelism(self, model: nn.Module, tp_config: Optional[Dict[str, Any]], pp_config: Optional[Dict[str, Any]]) -> nn.Module:
        """
        Apply PP, TP, and DP sequentially to the model.
        
        Application order: PP -> TP -> DP
        - PP (outermost): Splits model structure across pipeline stages
        - TP (middle): Splits individual layers across tensor parallel ranks  
        - DP (innermost): Handles gradient synchronization
        
        Args:
            model: Original model
            tp_config: TP configuration
            pp_config: PP configuration
            
        Returns:
            Model with parallelism applied
        """
        wrapped_model = model
        
        # Step 1: Apply Pipeline Parallelism (outermost)
        if self.pp_size > 1:
            if pp_config is None:
                raise ValueError("pp_config must be provided when PP size > 1")
            
            print(f"[Rank {self.parallel_context.rank}] Applying Pipeline Parallelism (PP={self.pp_size})")
            wrapped_model = PPWrapper(
                model=wrapped_model,
                parallel_context=self.parallel_context,
                block_names=pp_config.get("block_names", [])
            )
        
        # Step 2: Apply Tensor Parallelism (middle)
        if self.tp_size > 1:
            if tp_config is None:
                raise ValueError("tp_config must be provided when TP size > 1")
            
            print(f"[Rank {self.parallel_context.rank}] Applying Tensor Parallelism (TP={self.tp_size})")
            wrapped_model = TPWrapper(
                model=wrapped_model,
                parallel_context=self.parallel_context,
                column_linear_names=tp_config.get("column_linear_names", []),
                row_linear_names=tp_config.get("row_linear_names", []),
                auto_tune=self.auto_tune
            )
        
        # Step 3: Apply Data Parallelism (innermost)
        if self.dp_size > 1:
            print(f"[Rank {self.parallel_context.rank}] Applying Data Parallelism (DP={self.dp_size})")
            wrapped_model = DPWrapper(
                model=wrapped_model,
                parallel_context=self.parallel_context,
                auto_tune=self.auto_tune
            )
        
        return wrapped_model
    
    def forward(self, *args, **kwargs):
        """Forward pass through the hybrid parallel model."""
        # Handle DP gradient synchronization
        if self.dp_size > 1 and self.require_backward_grad_sync:
            self.enable_grad_sync()
            self.require_backward_grad_sync = False
        
        return self.model(*args, **kwargs)
    
    def enable_grad_sync(self):
        """Enable gradient synchronization for DP."""
        if self.dp_size > 1:
            # If we have a DPWrapper, enable its grad sync
            if hasattr(self.model, 'enable_grad_sync'):
                self.model.enable_grad_sync()
            else:
                # Fallback: manually set bwd_sync on parameters
                for param in self.model.parameters():
                    if hasattr(param, 'bwd_sync'):
                        setattr(param, 'bwd_sync', True)
    
    def get_parallel_info(self):
        """Get information about the parallel configuration."""
        info = {
            'world_size': self.parallel_context.world_size,
            'global_rank': self.parallel_context.rank,
            'coordinates': self.parallel_context.get_coord_dict(),
            'tp_size': self.tp_size,
            'dp_size': self.dp_size,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Add TP-specific info if available
        if hasattr(self.model, 'get_tp_info'):
            info['tp_info'] = self.model.get_tp_info()
        
        return info
    
    def print_parallel_info(self):
        """Print parallel configuration information."""
        info = self.get_parallel_info()
        rank = info['global_rank']
        
        print(f"\n[Rank {rank}] Hybrid Parallel Configuration:")
        print(f"  Global: rank={rank}/{info['world_size']}")
        print(f"  Coordinates: {info['coordinates']}")
        print(f"  TP size: {info['tp_size']}, DP size: {info['dp_size']}")
        print(f"  Total params: {info['total_parameters']:,}")
        print(f"  Trainable params: {info['trainable_parameters']:,}")
        
        if 'tp_info' in info:
            tp_info = info['tp_info']
            print(f"  TP rank: {tp_info['tp_rank']}")
            print(f"  TP modules: col={tp_info['column_parallel_modules']}, row={tp_info['row_parallel_modules']}")


def apply_hybrid_parallel(model: nn.Module,
                         parallel_context: ParallelContext,
                         column_linear_names: Optional[List[str]] = None,
                         row_linear_names: Optional[List[str]] = None,
                         block_names: Optional[List[str]] = None,
                         auto_tune: bool = False) -> HybridParallelWrapper:
    """
    Apply hybrid parallelism (TP + DP + PP) to a model.
    
    Args:
        model (nn.Module): The model to apply hybrid parallelism to
        parallel_context (ParallelContext): The parallel context with TP, DP, and PP configuration
        column_linear_names (List[str], optional): Names of modules for column parallel (TP)
        row_linear_names (List[str], optional): Names of modules for row parallel (TP)
        block_names (List[str], optional): Names of block containers for pipeline parallel (PP)
        auto_tune (bool): Whether to enable auto tuning
        
    Returns:
        HybridParallelWrapper: The wrapped model with hybrid parallelism applied
        
    Example:
        >>> # Initialize distributed (8 GPUs: TP=2, DP=2, PP=2)
        >>> dist.init_process_group(backend="nccl", init_method="env://")
        >>> ctx = ParallelContext({"tp": 2, "dp": 2, "pp": 2})
        >>> 
        >>> # Apply 3D hybrid parallelism
        >>> model = GPT2Model(config)
        >>> hybrid_model = apply_hybrid_parallel(
        ...     model=model,
        ...     parallel_context=ctx,
        ...     column_linear_names=["c_attn"],
        ...     row_linear_names=["c_proj"],
        ...     block_names=["transformer.h"],
        ...     auto_tune=True
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     hybrid_model.require_backward_grad_sync = True  # Enable DP grad sync
        ...     loss = hybrid_model(batch)
        ...     loss.backward()  # PP+TP communication automatic, DP grad sync enabled
        ...     optimizer.step()
    """
    # Prepare TP configuration
    tp_config = None
    tp_size = parallel_context.parallel_dims.get("tp", 1)
    
    if tp_size > 1:
        tp_config = {
            "column_linear_names": column_linear_names or [],
            "row_linear_names": row_linear_names or []
        }
    
    # Prepare PP configuration
    pp_config = None
    pp_size = parallel_context.parallel_dims.get("pp", 1)
    
    if pp_size > 1:
        pp_config = {
            "block_names": block_names or []
        }
    
    return HybridParallelWrapper(
        model=model,
        parallel_context=parallel_context,
        tp_config=tp_config,
        pp_config=pp_config,
        auto_tune=auto_tune
    )


# Convenient aliases
HP = HybridParallelWrapper
apply_2d_parallel = apply_hybrid_parallel 