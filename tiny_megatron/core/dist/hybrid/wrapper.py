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
    Unified Hybrid Parallel Wrapper supporting 2D parallelism: TP + DP.
    
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
                 dp_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Unified Hybrid Parallel wrapper.
        
        Args:
            model (nn.Module): The original model to wrap
            parallel_context (ParallelContext): The parallel context containing TP and DP configs
            tp_config (Dict, optional): TP configuration containing:
                - column_linear_patterns (List[str]): Path patterns for column parallel modules
                - row_linear_patterns (List[str]): Path patterns for row parallel modules
            dp_config (Dict, optional): DP configuration (reserved for future use)
        """
        super().__init__()
        
        if parallel_context is None:
            raise ValueError("parallel_context must be provided for Unified Hybrid wrapper")
        
        self.parallel_context = parallel_context
        
        # Extract parallel sizes from parallel context
        self.tp_size = parallel_context.parallel_dims.get("tp", 1)
        self.dp_size = parallel_context.parallel_dims.get("dp", 1)
        
        # Get ranks
        self.tp_rank = parallel_context.get_rank_in("tp") if self.tp_size > 1 else 0
        self.dp_rank = parallel_context.get_rank_in("dp") if self.dp_size > 1 else 0
        
        # Validate configuration
        total_parallel_dims = sum([1 for size in [self.tp_size, self.dp_size] if size > 1])
        if total_parallel_dims == 0:
            raise ValueError("At least one of TP or DP must have size > 1 for hybrid parallelism")
        
        # Store configurations
        self.tp_config = tp_config or {}
        self.dp_config = dp_config or {}
        
        # Set target device for hybrid parameters based on rank
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{parallel_context.rank}")
        else:
            self.device = torch.device("cpu")
        
        # Create hybrid model
        self.model = self._create_hybrid_model(model)
        
        # Move model to target device
        self.model = self.model.to(self.device)
        
        # Initialize DP state for gradient synchronization
        self.require_dp_backward_grad_sync = False
        
        print(f"[Rank {parallel_context.rank}] HybridParallelWrapper initialized:")
        print(f"  - TP size: {self.tp_size}, DP size: {self.dp_size}")
        print(f"  - Coordinates: {parallel_context.get_coord_dict()}")

    def _create_hybrid_model(self, model: nn.Module) -> nn.Module:
        """
        Create a hybrid model by replacing compatible modules with unified hybrid versions.
        
        This method walks through the model and replaces Linear, LayerNorm, and Embedding 
        modules with their hybrid equivalents that can handle TP and DP.
        """
        # Create a deep copy of the model
        model_copy = copy.deepcopy(model)
        
        # Get TP patterns for module classification
        column_linear_patterns = self.tp_config.get("column_linear_patterns", [])
        row_linear_patterns = self.tp_config.get("row_linear_patterns", [])
        
        # Replace modules with hybrid versions
        self._replace_modules_recursive(
            model_copy, "", column_linear_patterns, row_linear_patterns)
        
        return model_copy

    def _replace_modules_recursive(self, 
                                 module: nn.Module, 
                                 name_prefix: str,
                                 column_linear_patterns: List[str],
                                 row_linear_patterns: List[str]):
        """
        Recursively replace modules with hybrid versions.
        
        Args:
            module: Current module being processed
            name_prefix: Path prefix for current module
            column_linear_patterns: List of patterns for column parallel Linear modules
            row_linear_patterns: List of patterns for row parallel Linear modules
        """
        for name, child_module in list(module.named_children()):
            full_path = f"{name_prefix}.{name}" if name_prefix else name
            
            # Check if this is a Linear module
            if isinstance(child_module, nn.Linear):
                tp_mode = self._classify_tp_mode(full_path, column_linear_patterns, row_linear_patterns)
                
                if tp_mode != "none":
                    new_module = self._create_hybrid_linear(child_module, tp_mode)
                    setattr(module, name, new_module)
                    print(f"Replaced {full_path} with HybridLinear (tp_mode={tp_mode})")
            
            # Check if this is a LayerNorm module
            elif isinstance(child_module, nn.LayerNorm):
                # LayerNorm can still be wrapped for future DP optimizations
                new_module = self._create_hybrid_layernorm(child_module)
                setattr(module, name, new_module)
                print(f"Replaced {full_path} with HybridLayerNorm")
            
            # Check if this is an Embedding module  
            elif isinstance(child_module, nn.Embedding):
                tp_enabled = self._should_enable_tp_for_embedding(full_path)
                if tp_enabled:
                    new_module = self._create_hybrid_embedding(child_module, tp_enabled)
                    setattr(module, name, new_module)
                    print(f"Replaced {full_path} with HybridEmbedding (tp_enabled={tp_enabled})")
            
            # Recursively process child modules
            self._replace_modules_recursive(
                child_module, full_path, column_linear_patterns, row_linear_patterns)

    def _classify_tp_mode(self, module_path: str, column_patterns: List[str], row_patterns: List[str]) -> str:
        """Classify TP mode for a Linear module based on path patterns."""
        for pattern in column_patterns:
            if pattern in module_path:
                return "column"
        
        for pattern in row_patterns:
            if pattern in module_path:
                return "row"
        
        return "none"

    def _should_enable_tp_for_embedding(self, module_path: str) -> bool:
        """Check if TP should be enabled for an embedding module."""
        return self.tp_size > 1 and ("wte" in module_path or "embedding" in module_path.lower())

    def _create_hybrid_linear(self, original_module: nn.Linear, tp_mode: str) -> HybridLinear:
        """Create a HybridLinear module from original Linear module."""
        # Get original device and dtype
        device = original_module.weight.device
        dtype = original_module.weight.dtype
        
        # Create hybrid module with correct parameters
        hybrid_module = HybridLinear(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=original_module.bias is not None,
            device=device,
            dtype=dtype,
            auto_tune=False,  # Can be made configurable
            parallel_context=self.parallel_context,
            tp_mode=tp_mode,
            dp_enabled=self.dp_size > 1
        )
        
        # Transfer state dict with proper weight sharding for TP
        self._transfer_linear_state_dict(hybrid_module, original_module, tp_mode)
        
        return hybrid_module

    def _create_hybrid_layernorm(self, original_module: nn.Module) -> HybridLayerNorm:
        """Create a HybridLayerNorm module from original LayerNorm module."""
        # Handle both standard nn.LayerNorm and custom LayerNorm
        if hasattr(original_module, 'normalized_shape'):
            normalized_shape = original_module.normalized_shape
        else:
            # For custom LayerNorm, infer from weight shape
            normalized_shape = original_module.weight.shape if original_module.weight is not None else (768,)
        
        # Get original device and dtype
        device = original_module.weight.device if original_module.weight is not None else None
        dtype = original_module.weight.dtype if original_module.weight is not None else None
        
        # Create hybrid module
        hybrid_module = HybridLayerNorm(
            normalized_shape=normalized_shape,
            eps=getattr(original_module, 'eps', 1e-5),
            elementwise_affine=original_module.weight is not None,
            bias=original_module.bias is not None,
            device=device,
            dtype=dtype,
            auto_tune=False,  # Can be made configurable
            parallel_context=self.parallel_context
        )
        
        # Transfer state dict
        self._transfer_layernorm_state_dict(hybrid_module, original_module)
        
        return hybrid_module

    def _create_hybrid_embedding(self, original_module: nn.Embedding, tp_enabled: bool) -> HybridEmbedding:
        """Create a HybridEmbedding module from original Embedding module."""
        # Get original device and dtype
        device = original_module.weight.device
        dtype = original_module.weight.dtype
        
        # Create hybrid module
        hybrid_module = HybridEmbedding(
            num_embeddings=original_module.num_embeddings,
            embedding_dim=original_module.embedding_dim,
            padding_idx=original_module.padding_idx,
            max_norm=getattr(original_module, 'max_norm', None),
            norm_type=getattr(original_module, 'norm_type', 2.0),
            scale_grad_by_freq=getattr(original_module, 'scale_grad_by_freq', False),
            sparse=getattr(original_module, 'sparse', False),
            _weight=None,
            _freeze=False,
            device=device,
            dtype=dtype,
            auto_tune=False,  # Can be made configurable
            parallel_context=self.parallel_context,
            tp_enabled=tp_enabled
        )
        
        # Transfer state dict with proper weight sharding for TP
        self._transfer_embedding_state_dict(hybrid_module, original_module, tp_enabled)
        
        return hybrid_module
    
    def _transfer_linear_state_dict(self, hybrid_module: HybridLinear, original_module: nn.Linear, tp_mode: str):
        """Transfer and shard weights from original Linear to HybridLinear."""
        with torch.no_grad():
            if tp_mode == "column":
                # Column parallel: shard output dimension (dim=0)
                local_out_features = hybrid_module.out_features
                start_idx = hybrid_module.tp_rank * local_out_features
                end_idx = start_idx + local_out_features
                
                # Shard weight
                hybrid_module.weight.copy_(original_module.weight[start_idx:end_idx, :])
                
                # Shard bias if present
                if hybrid_module.bias is not None and original_module.bias is not None:
                    hybrid_module.bias.copy_(original_module.bias[start_idx:end_idx])
                    
            elif tp_mode == "row":
                # Row parallel: shard input dimension (dim=1)
                local_in_features = hybrid_module.in_features
                start_idx = hybrid_module.tp_rank * local_in_features
                end_idx = start_idx + local_in_features
                
                # Shard weight
                hybrid_module.weight.copy_(original_module.weight[:, start_idx:end_idx])
                
                # Only rank 0 gets bias for row parallel
                if hybrid_module.bias is not None and original_module.bias is not None:
                    hybrid_module.bias.copy_(original_module.bias)
                    
            else:
                # No TP: direct copy
                hybrid_module.weight.copy_(original_module.weight)
                if hybrid_module.bias is not None and original_module.bias is not None:
                    hybrid_module.bias.copy_(original_module.bias)
    
    def _transfer_layernorm_state_dict(self, hybrid_module: HybridLayerNorm, original_module: nn.Module):
        """Transfer weights from original LayerNorm to HybridLayerNorm."""
        with torch.no_grad():
            if hybrid_module.weight is not None and original_module.weight is not None:
                hybrid_module.weight.copy_(original_module.weight)
            if hybrid_module.bias is not None and original_module.bias is not None:
                hybrid_module.bias.copy_(original_module.bias)
    
    def _transfer_embedding_state_dict(self, hybrid_module: HybridEmbedding, original_module: nn.Embedding, tp_enabled: bool):
        """Transfer and shard weights from original Embedding to HybridEmbedding."""
        with torch.no_grad():
            if tp_enabled and hybrid_module.tp_size > 1:
                # TP enabled: shard embedding dimension
                local_embedding_dim = hybrid_module.embedding_dim
                start_idx = hybrid_module.tp_rank * local_embedding_dim
                end_idx = start_idx + local_embedding_dim
                
                # Shard embedding weight
                hybrid_module.weight.copy_(original_module.weight[:, start_idx:end_idx])
            else:
                # No TP: direct copy
                hybrid_module.weight.copy_(original_module.weight)

    def forward(self, *args, **kwargs):
        if self.require_dp_backward_grad_sync:
            self.enable_grad_sync()
        self.require_dp_backward_grad_sync = False
        return self.model(*args, **kwargs)

    def enable_grad_sync(self):
        for param in self.model.parameters():
            setattr(param, 'dp_bwd_sync', True)

    def get_hybrid_info(self):
        """Get information about the hybrid configuration."""
        info = {
            'tp_size': self.tp_size,
            'dp_size': self.dp_size,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device)
        }
        
        return info

    def print_hybrid_info(self):
        """Print hybrid parallel configuration information."""
        info = self.get_hybrid_info()
        print(f"\nHybrid Parallel Configuration:")
        print(f"  TP size: {info['tp_size']}, DP size: {info['dp_size']}")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        print(f"  Device: {info['device']}")


def apply_hybrid_parallel(model: nn.Module,
                         parallel_context: ParallelContext,
                         tp_config: Optional[Dict[str, Any]] = None,
                         dp_config: Optional[Dict[str, Any]] = None) -> HybridParallelWrapper:
    """
    Apply hybrid parallelism (TP + DP) to a model.
    
    Args:
        model (nn.Module): The model to apply hybrid parallelism to
        parallel_context (ParallelContext): The parallel context
        tp_config (Dict, optional): TP configuration
        dp_config (Dict, optional): DP configuration (reserved for future use)
        
    Returns:
        HybridParallelWrapper: The wrapped model with hybrid parallelism applied
        
    Example:
        >>> # Initialize distributed and create context
        >>> dist.init_process_group(backend="nccl", init_method="env://")
        >>> ctx = ParallelContext({"tp": 2, "dp": 2})
        >>> 
        >>> # Apply hybrid parallelism to model
        >>> model = GPT2Model(config)
        >>> tp_config = {
        ...     "column_linear_patterns": ["attn.c_attn", "mlp.c_fc"],
        ...     "row_linear_patterns": ["attn.c_proj", "mlp.c_proj"]
        ... }
        >>> hybrid_model = apply_hybrid_parallel(
        ...     model=model,
        ...     parallel_context=ctx,
        ...     tp_config=tp_config
        ... )
        >>> 
        >>> # Use the hybrid model
        >>> output = hybrid_model(input_ids)
    """
    # Set default TP configuration if TP is enabled
    if parallel_context.parallel_dims.get("tp", 1) > 1:
        default_tp_config = {
            "column_linear_patterns": ["attn.c_attn", "mlp.c_fc"],
            "row_linear_patterns": ["attn.c_proj", "mlp.c_proj"]
        }
        # Merge with user-provided config
        if tp_config:
            default_tp_config.update(tp_config)
        tp_config = default_tp_config
    
    return HybridParallelWrapper(
        model=model,
        parallel_context=parallel_context,
        tp_config=tp_config,
        dp_config=dp_config
    ) 