# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.distributed as dist
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
                - column_linear_patterns (List[str]): Path patterns for column parallel modules
                - row_linear_patterns (List[str]): Path patterns for row parallel modules
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
        self.tp_rank = parallel_context.get_rank_in("tp") if self.tp_size > 1 else 0
        self.dp_rank = parallel_context.get_rank_in("dp") if self.dp_size > 1 else 0
        self.pp_rank = parallel_context.get_rank_in("pp") if self.pp_size > 1 else 0
        
        # Calculate previous and next global ranks for PP communication
        if self.pp_size > 1:
            self.prev_rank = self._pp_rank_to_global_rank(self.pp_rank - 1) if self.pp_rank > 0 else None
            self.next_rank = self._pp_rank_to_global_rank(self.pp_rank + 1) if self.pp_rank < self.pp_size - 1 else None
            self.pp_group = parallel_context.get_group("pp")
        else:
            self.prev_rank = None
            self.next_rank = None
            self.pp_group = None
        
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
        
        # Set target device for hybrid parameters based on rank
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{parallel_context.rank}")
        else:
            self.device = torch.device("cpu")
        
        # Create hybrid model
        self.model = self._create_hybrid_model(model)
        
        # Move entire model to target device
        self.model = self.model.to(self.device)
        
        # Track gradient sync requirement for DP
        self.require_backward_grad_sync = False
        
        print(f"[Rank {parallel_context.rank}] HybridParallelWrapper initialized:")
        print(f"  - TP size: {self.tp_size}, DP size: {self.dp_size}, PP size: {self.pp_size}")
        print(f"  - Coordinates: {parallel_context.get_coord_dict()}")
    
    def _pp_rank_to_global_rank(self, pp_rank: int) -> int:
        """Convert PP rank to global rank."""
        coords = self.parallel_context.get_coord_dict()
        target_coords = coords.copy()
        target_coords['pp'] = pp_rank
        
        return self.parallel_context._coords_to_rank(
            [target_coords[name] for name in self.parallel_context.dim_names]
        )
    
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
        
        # Get module pattern lists from configurations
        column_linear_patterns = self.tp_config.get("column_linear_patterns", [])
        row_linear_patterns = self.tp_config.get("row_linear_patterns", [])
        block_names = self.pp_config.get("block_names", [])
        
        # Determine which blocks belong to this PP rank
        pp_owned_blocks = set()
        if self.pp_size > 1:
            pp_owned_blocks = self._get_pp_owned_blocks(model_copy, block_names)
        
        # Replace modules with hybrid versions
        self._replace_modules_recursive(
            model_copy, 
            "", 
            column_linear_patterns,
            row_linear_patterns,
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
                                   column_linear_patterns: List[str],
                                   row_linear_patterns: List[str], 
                                   pp_owned_blocks: set):
        """
        Recursively replace modules with hybrid versions.
        
        Args:
            module: Current module being processed
            module_path: Dot-separated path to current module
            column_linear_patterns: Path patterns for column parallel TP
            row_linear_patterns: Path patterns for row parallel TP
            pp_owned_blocks: Set of block paths owned by current PP rank
        """
        for child_name, child_module in list(module.named_children()):
            full_path = f"{module_path}.{child_name}" if module_path else child_name
            
            # Check if this module is in a PP-owned block
            pp_enabled = self._is_in_pp_owned_block(full_path, pp_owned_blocks)
            
            # Replace Linear modules
            if isinstance(child_module, nn.Linear):
                tp_mode = self._get_tp_mode(full_path, column_linear_patterns, row_linear_patterns)
                
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
                self._replace_modules_recursive(child_module, full_path, column_linear_patterns, 
                                              row_linear_patterns, pp_owned_blocks)
    
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
    
    def _get_tp_mode(self, module_path: str, column_patterns: List[str], row_patterns: List[str]) -> str:
        """
        Determine TP mode for a module based on its full path.
        
        Args:
            module_path: Full dot-separated path to the module
            column_patterns: List of path patterns for column parallel
            row_patterns: List of path patterns for row parallel
            
        Returns:
            TP mode: "none", "column", or "row"
        """
        if self.tp_size <= 1:
            return "none"
        
        # Check against column parallel patterns
        for pattern in column_patterns:
            if self._path_matches_pattern(module_path, pattern):
                return "column"
        
        # Check against row parallel patterns
        for pattern in row_patterns:
            if self._path_matches_pattern(module_path, pattern):
                return "row"
        
        return "none"
    
    def _path_matches_pattern(self, module_path: str, pattern: str) -> bool:
        """
        Check if a module path matches a pattern.
        
        Supports simple wildcard matching with '*' for any part.
        
        Args:
            module_path: Full path like "transformer.h.0.attn.c_attn"
            pattern: Pattern like "*.attn.c_attn" or "transformer.h.*.attn.c_proj"
            
        Returns:
            True if path matches pattern
        """
        path_parts = module_path.split('.')
        pattern_parts = pattern.split('.')
        
        # Simple wildcard matching
        if len(path_parts) != len(pattern_parts):
            return False
        
        for path_part, pattern_part in zip(path_parts, pattern_parts):
            if pattern_part != '*' and pattern_part != path_part:
                return False
        
        return True
    
    def _create_hybrid_linear(self, original_module: nn.Linear, tp_mode: str, pp_enabled: bool) -> HybridLinear:
        """Create a HybridLinear module from original Linear module."""
        # Get original device and dtype
        device = original_module.weight.device
        dtype = original_module.weight.dtype
        
        # Create hybrid module
        hybrid_module = HybridLinear(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=original_module.bias is not None,
            device=device,
            dtype=dtype,
            auto_tune=False,  # Can be made configurable
            parallel_context=self.parallel_context,
            tp_mode=tp_mode,
            pp_enabled=pp_enabled,
            dp_enabled=self.dp_size > 1
        )
        
        # Transfer state dict with proper weight sharding for TP
        self._transfer_linear_state_dict(hybrid_module, original_module, tp_mode)
        
        return hybrid_module
    
    def _create_hybrid_layernorm(self, original_module: nn.Module, pp_enabled: bool) -> HybridLayerNorm:
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
            parallel_context=self.parallel_context,
            pp_enabled=pp_enabled
        )
        
        # Transfer state dict
        self._transfer_layernorm_state_dict(hybrid_module, original_module)
        
        return hybrid_module
    
    def _create_hybrid_embedding(self, original_module: nn.Embedding, tp_enabled: bool, pp_enabled: bool) -> HybridEmbedding:
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
            device=device,
            dtype=dtype,
            auto_tune=False,  # Can be made configurable
            parallel_context=self.parallel_context,
            tp_enabled=tp_enabled,
            pp_enabled=pp_enabled
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
        """Forward pass through the unified hybrid model with PP orchestration."""
        # Handle DP gradient synchronization
        if self.dp_size > 1 and self.require_backward_grad_sync:
            self._enable_grad_sync()
            self.require_backward_grad_sync = False
        
        # Handle pipeline parallelism if enabled
        if self.pp_size > 1:
            return self._forward_with_pipeline(*args, **kwargs)
        else:
            # No PP: direct forward pass
            return self.model(*args, **kwargs)
    
    def _forward_with_pipeline(self, *args, **kwargs):
        """Forward pass with pipeline parallelism orchestration."""
        # Stage 0 (first): process initial layers and send to next stage
        if self.pp_rank == 0:
            # Process through assigned layers
            hidden_states = self._process_first_stage(*args, **kwargs)
            
            # Send to next stage if not single stage
            if self.pp_size > 1:
                self._send_to_next_stage(hidden_states)
            else:
                # Single stage - process everything (shouldn't reach here in PP mode)
                return self.model(*args, **kwargs)
            
            # For multi-stage, first stage doesn't return final output
            return None, None
            
        # Stage N-1 (last): receive from previous, process final layers
        elif self.pp_rank == self.pp_size - 1:
            # Receive hidden states from previous stage
            hidden_states = self._recv_from_prev_stage()
            
            # Process final layers and compute loss
            return self._process_last_stage(hidden_states, *args, **kwargs)
            
        # Middle stages: receive, process assigned blocks, send
        else:
            # Receive from previous stage
            hidden_states = self._recv_from_prev_stage()
            
            # Process through assigned blocks
            output = self._process_middle_stage(hidden_states)
            
            # Send to next stage
            self._send_to_next_stage(output)
            
            # Middle stages don't return final output
            return None, None
    
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


    def _send_to_next_stage(self, tensor: torch.Tensor) -> None:
        """Send tensor to next PP stage."""
        if self.next_rank is not None:
            dist.send(tensor.contiguous(), dst=self.next_rank, group=self.pp_group)
    
    def _recv_from_prev_stage(self) -> torch.Tensor:
        """Receive tensor from previous PP stage."""
        if self.prev_rank is None:
            raise RuntimeError("Cannot receive from previous stage: prev_rank is None")
            
        # Create tensor with appropriate shape (hardcoded for now)
        # In practice, this would be negotiated or pre-agreed
        batch_size, seq_len, hidden_size = 1, 1024, 768
        tensor = torch.empty(batch_size, seq_len, hidden_size, 
                           dtype=torch.float32, device=self.device)
        
        dist.recv(tensor, src=self.prev_rank, group=self.pp_group)
        return tensor
    
    def _process_first_stage(self, *args, **kwargs):
        """Process the first stage of the pipeline."""
        # For first stage, run the model up to the first cut point
        # This is a simplified implementation - in practice, would need to
        # identify where to cut the model based on block_names
        hidden_states = self.model(*args, **kwargs)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]  # Take hidden states, ignore loss
        return hidden_states
    
    def _process_middle_stage(self, hidden_states):
        """Process middle stage of the pipeline."""
        # Run model with received hidden states
        # This is simplified - would need proper input handling
        return self.model(hidden_states)
    
    def _process_last_stage(self, hidden_states, *args, **kwargs):
        """Process the last stage of the pipeline."""
        # Run final layers and compute loss if target is provided
        logits = self.model(hidden_states)
        
        # Compute loss if target is provided
        if len(args) > 1:  # Assuming target is the second argument
            target = args[1]
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
            return logits, loss
        else:
            return logits, None


def apply_hybrid_parallel(model: nn.Module,
                                  parallel_context: ParallelContext,
                                  column_linear_patterns: Optional[List[str]] = None,
                                  row_linear_patterns: Optional[List[str]] = None,
                                  block_names: Optional[List[str]] = None) -> HybridParallelWrapper:
    """
    Apply unified hybrid parallelism (TP + DP + PP) to a model.
    
    Args:
        model (nn.Module): The model to apply hybrid parallelism to
        parallel_context (ParallelContext): The parallel context with TP, DP, and PP configuration
        column_linear_patterns (List[str], optional): Path patterns for column parallel (TP)
            Example: ["*.attn.c_attn"] matches transformer.h.0.attn.c_attn, transformer.h.1.attn.c_attn, etc.
        row_linear_patterns (List[str], optional): Path patterns for row parallel (TP)
            Example: ["*.attn.c_proj", "*.mlp.c_proj"] matches both attention and MLP projections
        block_names (List[str], optional): Names of block containers for pipeline parallel (PP)
        
    Returns:
        HybridParallelWrapper: The wrapped model with unified hybrid parallelism applied
        
    Example:
        >>> # Initialize distributed (8 GPUs: TP=2, DP=2, PP=2)
        >>> dist.init_process_group(backend="nccl", init_method="env://")
        >>> ctx = ParallelContext({"tp": 2, "dp": 2, "pp": 2})
        >>> 
        >>> # Apply unified 3D hybrid parallelism
        >>> model = GPT2Model(config)
        >>> hybrid_model = apply_hybrid_parallel(
        ...     model=model,
        ...     parallel_context=ctx,
        ...     column_linear_patterns=["*.attn.c_attn"],  # QKV projection
        ...     row_linear_patterns=["*.attn.c_proj", "*.mlp.c_proj"],  # Output projections
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
            "column_linear_patterns": column_linear_patterns or [],
            "row_linear_patterns": row_linear_patterns or []
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