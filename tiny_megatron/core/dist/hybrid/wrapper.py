# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import copy
from typing import Optional, Dict, List, Any
from ..utils.comm import ParallelContext
from .modules import HybridLinear, HybridLayerNorm, HybridEmbedding


class HybridParallelWrapper(nn.Module):
    """
    Unified Hybrid Parallel Wrapper supporting 3D parallelism: TP + DP + PP.
    
    Unlike the original HybridParallelWrapper that chains multiple wrappers,
    this version directly replaces modules with unified hybrid modules that
    handle all parallelism strategies internally.
    
    This approach avoids module replacement conflicts and provides better
    performance by handling all communication in a single forward pass.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 parallel_context: ParallelContext,
                 tp_config: Optional[Dict[str, Any]] = None,
                 pp_config: Optional[Dict[str, Any]] = None,
                 dp_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Unified Hybrid Parallel wrapper.
        
        Args:
            model (nn.Module): The original model to wrap
            parallel_context (ParallelContext): The parallel context containing TP, DP, and PP configs
            tp_config (Dict, optional): TP configuration containing:
                - column_linear_names (List[str]): Names of modules for column parallel
                - row_linear_names (List[str]): Names of modules for row parallel  
                - qkv_column_names (List[str]): Names of QKV column parallel modules
                - qkv_row_names (List[str]): Names of QKV row parallel modules
            pp_config (Dict, optional): PP configuration containing:
                - block_names (List[str]): Names of block containers to distribute across PP ranks
            dp_config (Dict, optional): DP configuration (reserved for future use)
        """
        super().__init__()
        
        if parallel_context is None:
            raise ValueError("parallel_context must be provided for Unified Hybrid wrapper")
        
        self.parallel_context = parallel_context
        
        # Extract parallel sizes from parallel context
        self.tp_size = parallel_context.parallel_dims.get("tp", 1)
        self.dp_size = parallel_context.parallel_dims.get("dp", 1)
        self.pp_size = parallel_context.parallel_dims.get("pp", 1)
        
        # Get ranks
        self.tp_rank = parallel_context.get_coord("tp") if self.tp_size > 1 else 0
        self.dp_rank = parallel_context.get_coord("dp") if self.dp_size > 1 else 0
        self.pp_rank = parallel_context.get_coord("pp") if self.pp_size > 1 else 0
        
        # Validate configuration
        total_parallel_dims = sum([1 for size in [self.tp_size, self.dp_size, self.pp_size] if size > 1])
        if total_parallel_dims == 0:
            raise ValueError("At least one of TP, DP, or PP must have size > 1 for hybrid parallelism")
        
        # Store original model reference
        self.original_model = model
        
        # Store configurations
        self.tp_config = tp_config or {}
        self.pp_config = pp_config or {}
        self.dp_config = dp_config or {}
        
        # Create hybrid model
        self.model = self._create_hybrid_model(model)
        
        # Track gradient sync requirement for DP
        self.require_backward_grad_sync = False
        
        print(f"[Rank {parallel_context.rank}] UnifiedHybridWrapper initialized:")
        print(f"  - TP size: {self.tp_size}, DP size: {self.dp_size}, PP size: {self.pp_size}")
        print(f"  - Coordinates: {parallel_context.get_coord_dict()}")
    
    def _create_hybrid_model(self, model: nn.Module) -> nn.Module:
        """
        Create hybrid model by replacing modules with unified hybrid versions.
        
        Args:
            model: Original model
            
        Returns:
            Model with hybrid modules
        """
        # Create a deep copy of the model
        model_copy = copy.deepcopy(model)
        
        # Get module name lists from configurations
        column_linear_names = self.tp_config.get("column_linear_names", [])
        row_linear_names = self.tp_config.get("row_linear_names", [])
        qkv_column_names = self.tp_config.get("qkv_column_names", [])
        qkv_row_names = self.tp_config.get("qkv_row_names", [])
        block_names = self.pp_config.get("block_names", [])
        
        # Determine which blocks belong to this PP rank
        pp_owned_blocks = set()
        if self.pp_size > 1:
            pp_owned_blocks = self._get_pp_owned_blocks(model_copy, block_names)
        
        # Replace modules with hybrid versions
        self._replace_modules_recursive(
            model_copy, 
            "", 
            column_linear_names,
            row_linear_names,
            qkv_column_names,
            qkv_row_names,
            pp_owned_blocks
        )
        
        return model_copy
    
    def _get_pp_owned_blocks(self, model: nn.Module, block_names: List[str]) -> set:
        """
        Determine which blocks belong to the current PP rank.
        
        Args:
            model: Model to analyze
            block_names: List of block container names
            
        Returns:
            Set of block paths owned by current PP rank
        """
        owned_blocks = set()
        
        for block_name in block_names:
            # Navigate to the block container
            container = model
            container_path = block_name.split('.')
            
            for attr in container_path[:-1]:
                container = getattr(container, attr)
            
            # Get the actual block container
            block_container = getattr(container, container_path[-1])
            
            if isinstance(block_container, nn.ModuleList):
                # Determine block distribution
                total_blocks = len(block_container)
                blocks_per_rank = total_blocks // self.pp_size
                start_idx = self.pp_rank * blocks_per_rank
                end_idx = start_idx + blocks_per_rank
                
                # Last rank takes any remaining blocks
                if self.pp_rank == self.pp_size - 1:
                    end_idx = total_blocks
                
                print(f"[Rank {self.parallel_context.rank}] PP blocks {block_name}: {start_idx}-{end_idx-1}")
                
                # Add owned block paths
                for i in range(start_idx, end_idx):
                    owned_blocks.add(f"{block_name}.{i}")
        
        return owned_blocks
    
    def _replace_modules_recursive(self, 
                                   module: nn.Module, 
                                   module_path: str,
                                   column_linear_names: List[str],
                                   row_linear_names: List[str], 
                                   qkv_column_names: List[str],
                                   qkv_row_names: List[str],
                                   pp_owned_blocks: set):
        """
        Recursively replace modules with hybrid versions.
        
        Args:
            module: Current module being processed
            module_path: Dot-separated path to current module
            column_linear_names: Names for column parallel TP
            row_linear_names: Names for row parallel TP
            qkv_column_names: Names for QKV column parallel TP
            qkv_row_names: Names for QKV row parallel TP
            pp_owned_blocks: Set of block paths owned by current PP rank
        """
        for child_name, child_module in list(module.named_children()):
            full_path = f"{module_path}.{child_name}" if module_path else child_name
            
            # Check if this module is in a PP-owned block
            pp_enabled = self._is_in_pp_owned_block(full_path, pp_owned_blocks)
            
            # Replace Linear modules
            if isinstance(child_module, nn.Linear):
                tp_mode = self._get_tp_mode(child_name, column_linear_names, row_linear_names, 
                                           qkv_column_names, qkv_row_names)
                
                if tp_mode != "none" or pp_enabled:
                    new_module = self._create_hybrid_linear(child_module, tp_mode, pp_enabled)
                    setattr(module, child_name, new_module)
                    print(f"Replaced {full_path} with HybridLinear (tp_mode={tp_mode}, pp_enabled={pp_enabled})")
            
            # Replace LayerNorm modules
            elif isinstance(child_module, (nn.LayerNorm, self._get_custom_layernorm_class())):
                if pp_enabled:
                    new_module = self._create_hybrid_layernorm(child_module, pp_enabled)
                    setattr(module, child_name, new_module)
                    print(f"Replaced {full_path} with HybridLayerNorm (pp_enabled={pp_enabled})")
            
            # Replace Embedding modules
            elif isinstance(child_module, nn.Embedding):
                tp_enabled = self.tp_size > 1
                if tp_enabled or pp_enabled:
                    new_module = self._create_hybrid_embedding(child_module, tp_enabled, pp_enabled)
                    setattr(module, child_name, new_module)
                    print(f"Replaced {full_path} with HybridEmbedding (tp_enabled={tp_enabled}, pp_enabled={pp_enabled})")
            
            # For non-PP-owned blocks, replace with Identity to save memory
            elif self.pp_size > 1 and not pp_enabled and self._is_block_module(full_path, pp_owned_blocks):
                setattr(module, child_name, nn.Identity())
                print(f"Replaced {full_path} with Identity (not owned by this PP rank)")
            
            else:
                # Recursively process children
                self._replace_modules_recursive(child_module, full_path, column_linear_names, 
                                              row_linear_names, qkv_column_names, qkv_row_names, 
                                              pp_owned_blocks)
    
    def _get_custom_layernorm_class(self):
        """Get custom LayerNorm class if available."""
        try:
            from ...module.normalization import LayerNorm
            return LayerNorm
        except ImportError:
            return type(None)  # Return a type that will never match
    
    def _is_in_pp_owned_block(self, module_path: str, pp_owned_blocks: set) -> bool:
        """Check if a module path is within a PP-owned block."""
        if self.pp_size <= 1:
            return True  # No PP, everything is "owned"
        
        for owned_block in pp_owned_blocks:
            if module_path.startswith(owned_block):
                return True
        return False
    
    def _is_block_module(self, module_path: str, pp_owned_blocks: set) -> bool:
        """Check if a module path represents a complete block."""
        for block_pattern in self.pp_config.get("block_names", []):
            # Extract block indices from owned blocks
            for owned_block in pp_owned_blocks:
                if module_path == owned_block:
                    return True
        
        # Also check if this path matches the pattern for non-owned blocks
        for block_pattern in self.pp_config.get("block_names", []):
            if module_path.startswith(block_pattern):
                # Extract the index part
                remainder = module_path[len(block_pattern):] 
                if remainder.startswith('.') and remainder[1:].isdigit():
                    return True
        
        return False
    
    def _get_tp_mode(self, module_name: str, column_names: List[str], row_names: List[str],
                     qkv_column_names: List[str], qkv_row_names: List[str]) -> str:
        """Determine TP mode for a module."""
        if self.tp_size <= 1:
            return "none"
        
        if module_name in qkv_column_names:
            return "qkv_column"
        elif module_name in qkv_row_names:
            return "qkv_row"
        elif module_name in column_names:
            return "column"
        elif module_name in row_names:
            return "row"
        else:
            return "none"
    
    def _create_hybrid_linear(self, original_module: nn.Linear, tp_mode: str, pp_enabled: bool) -> HybridLinear:
        """Create a HybridLinear module from original Linear module."""
        return HybridLinear(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=original_module.bias is not None,
            parallel_context=self.parallel_context,
            tp_mode=tp_mode,
            pp_enabled=pp_enabled,
            dp_enabled=self.dp_size > 1,
            device=original_module.weight.device,
            dtype=original_module.weight.dtype
        )
    
    def _create_hybrid_layernorm(self, original_module: nn.Module, pp_enabled: bool) -> HybridLayerNorm:
        """Create a HybridLayerNorm module from original LayerNorm module."""
        # Handle both standard nn.LayerNorm and custom LayerNorm
        if hasattr(original_module, 'normalized_shape'):
            normalized_shape = original_module.normalized_shape
        else:
            # For custom LayerNorm, infer from weight shape
            normalized_shape = original_module.weight.shape
        
        return HybridLayerNorm(
            normalized_shape=normalized_shape,
            eps=getattr(original_module, 'eps', 1e-5),
            elementwise_affine=original_module.weight is not None,
            bias=original_module.bias is not None,
            parallel_context=self.parallel_context,
            pp_enabled=pp_enabled,
            device=original_module.weight.device if original_module.weight is not None else None,
            dtype=original_module.weight.dtype if original_module.weight is not None else None
        )
    
    def _create_hybrid_embedding(self, original_module: nn.Embedding, tp_enabled: bool, pp_enabled: bool) -> HybridEmbedding:
        """Create a HybridEmbedding module from original Embedding module."""
        return HybridEmbedding(
            num_embeddings=original_module.num_embeddings,
            embedding_dim=original_module.embedding_dim,
            padding_idx=original_module.padding_idx,
            parallel_context=self.parallel_context,
            tp_enabled=tp_enabled,
            pp_enabled=pp_enabled,
            device=original_module.weight.device,
            dtype=original_module.weight.dtype
        )
    
    def forward(self, *args, **kwargs):
        """Forward pass through the unified hybrid model."""
        # Handle DP gradient synchronization
        if self.dp_size > 1 and self.require_backward_grad_sync:
            self._enable_grad_sync()
            self.require_backward_grad_sync = False
        
        return self.model(*args, **kwargs)
    
    def _enable_grad_sync(self):
        """Enable gradient synchronization for DP."""
        if self.dp_size > 1:
            # Set gradient sync flag on all parameters
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
            'pp_size': self.pp_size,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        return info
    
    def print_parallel_info(self):
        """Print parallel configuration information."""
        info = self.get_parallel_info()
        rank = info['global_rank']
        
        print(f"\n[Rank {rank}] Unified Hybrid Parallel Configuration:")
        print(f"  Global: rank={rank}/{info['world_size']}")
        print(f"  Coordinates: {info['coordinates']}")
        print(f"  TP size: {info['tp_size']}, DP size: {info['dp_size']}, PP size: {info['pp_size']}")
        print(f"  Total params: {info['total_parameters']:,}")
        print(f"  Trainable params: {info['trainable_parameters']:,}")


def apply_hybrid_parallel(model: nn.Module,
                                  parallel_context: ParallelContext,
                                  column_linear_names: Optional[List[str]] = None,
                                  row_linear_names: Optional[List[str]] = None,
                                  qkv_column_names: Optional[List[str]] = None,
                                  qkv_row_names: Optional[List[str]] = None,
                                  block_names: Optional[List[str]] = None) -> HybridParallelWrapper:
    """
    Apply unified hybrid parallelism (TP + DP + PP) to a model.
    
    Args:
        model (nn.Module): The model to apply hybrid parallelism to
        parallel_context (ParallelContext): The parallel context with TP, DP, and PP configuration
        column_linear_names (List[str], optional): Names of modules for column parallel (TP)
        row_linear_names (List[str], optional): Names of modules for row parallel (TP)
        qkv_column_names (List[str], optional): Names of modules for QKV column parallel (TP)
        qkv_row_names (List[str], optional): Names of modules for QKV row parallel (TP)
        block_names (List[str], optional): Names of block containers for pipeline parallel (PP)
        
    Returns:
        UnifiedHybridWrapper: The wrapped model with unified hybrid parallelism applied
        
    Example:
        >>> # Initialize distributed (8 GPUs: TP=2, DP=2, PP=2)
        >>> dist.init_process_group(backend="nccl", init_method="env://")
        >>> ctx = ParallelContext({"tp": 2, "dp": 2, "pp": 2})
        >>> 
        >>> # Apply unified 3D hybrid parallelism
        >>> model = GPT2Model(config)
        >>> hybrid_model = apply_unified_hybrid_parallel(
        ...     model=model,
        ...     parallel_context=ctx,
        ...     qkv_column_names=["c_attn"],
        ...     qkv_row_names=["c_proj"],
        ...     block_names=["transformer.h"]
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     hybrid_model.require_backward_grad_sync = True  # Enable DP grad sync
        ...     loss = hybrid_model(batch)  # All parallelism handled internally
        ...     loss.backward()
        ...     optimizer.step()
    """
    # Prepare TP configuration
    tp_config = {}
    tp_size = parallel_context.parallel_dims.get("tp", 1)
    
    if tp_size > 1:
        tp_config = {
            "column_linear_names": column_linear_names or [],
            "row_linear_names": row_linear_names or [],
            "qkv_column_names": qkv_column_names or [],
            "qkv_row_names": qkv_row_names or []
        }
    
    # Prepare PP configuration
    pp_config = {}
    pp_size = parallel_context.parallel_dims.get("pp", 1)
    
    if pp_size > 1:
        pp_config = {
            "block_names": block_names or []
        }
    
    # Prepare DP configuration (reserved for future use)
    dp_config = {}
    
    return HybridParallelWrapper(
        model=model,
        parallel_context=parallel_context,
        tp_config=tp_config,
        pp_config=pp_config,
        dp_config=dp_config
    ) 