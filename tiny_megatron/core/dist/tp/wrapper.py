# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional

from .module import (
    ColumnParallelLinear,
    RowParallelLinear
)
from ..utils.wrapper import get_init_args
from ..utils.comm import ParallelContext


class TPWrapper(nn.Module):
    """
    Tensor Parallel Wrapper for converting regular modules to TP modules.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 parallel_context: ParallelContext,
                 column_linear_names: Optional[List[str]] = None,
                 row_linear_names: Optional[List[str]] = None,
                 auto_tune: bool = False):
        """
        Initialize the TP wrapper.
        
        Args:
            model (nn.Module): The original model to wrap
            parallel_context (ParallelContext): The parallel context for TP communication
            column_linear_names (List[str], optional): Names of modules to replace with ColumnParallelLinear
            row_linear_names (List[str], optional): Names of modules to replace with RowParallelLinear  
            auto_tune (bool): Whether to enable auto tuning
        """
        super().__init__()
        
        if parallel_context is None:
            raise ValueError("parallel_context must be provided for TP wrapper")
            
        self.model = model
        self.parallel_context = parallel_context
        self.column_linear_names = column_linear_names or []
        self.row_linear_names = row_linear_names or []
        self.auto_tune = auto_tune
        
        # Validate that no module name appears in both lists
        overlapping = set(self.column_linear_names) & set(self.row_linear_names)
        if overlapping:
            raise ValueError(f"Module names cannot appear in both column and row lists: {overlapping}")
        
        # Apply TP wrapping
        self._wrap_layers()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)
    
    def _wrap_layers(self):
        """
        Wrap the selected layers with appropriate TP modules.
        """
        def _replace_module_recursive(module, path=''):
            for child_name, child in module.named_children():
                full_name = f"{path}.{child_name}" if path else child_name
                
                # Check if this module should be replaced
                if child_name in self.column_linear_names:
                    self._replace_with_column_parallel(module, child_name, child)
                elif child_name in self.row_linear_names:
                    self._replace_with_row_parallel(module, child_name, child)
                else:
                    # Recursively process children
                    _replace_module_recursive(child, full_name)
        
        _replace_module_recursive(self.model)
    
    def _replace_with_column_parallel(self, parent_module, module_name, original_module):
        """Replace a module with ColumnParallelLinear."""
        if not isinstance(original_module, nn.Linear):
            raise TypeError(f"Module {module_name} is not nn.Linear, cannot replace with ColumnParallelLinear")
        
        # Get initialization arguments
        init_args = get_init_args(original_module)
        
        # Create new ColumnParallelLinear module
        new_module = ColumnParallelLinear(
            **init_args,
            parallel_context=self.parallel_context,
            auto_tune=self.auto_tune
        )
        
        # Transfer device and training state
        device = next(original_module.parameters()).device
        new_module = new_module.to(device)
        new_module.train(original_module.training)
        
        # Load state dict (need to handle weight sharding)
        self._load_sharded_state_dict(new_module, original_module, shard_dim=0)
        
        # Replace the module
        setattr(parent_module, module_name, new_module)
        
        print(f"Replaced {module_name} with ColumnParallelLinear")
    
    def _replace_with_row_parallel(self, parent_module, module_name, original_module):
        """Replace a module with RowParallelLinear."""
        if not isinstance(original_module, nn.Linear):
            raise TypeError(f"Module {module_name} is not nn.Linear, cannot replace with RowParallelLinear")
        
        # Get initialization arguments  
        init_args = get_init_args(original_module)
        
        # Create new RowParallelLinear module
        new_module = RowParallelLinear(
            **init_args,
            parallel_context=self.parallel_context,
            auto_tune=self.auto_tune
        )
        
        # Transfer device and training state
        device = next(original_module.parameters()).device
        new_module = new_module.to(device)
        new_module.train(original_module.training)
        
        # Load state dict (need to handle weight sharding)
        self._load_sharded_state_dict(new_module, original_module, shard_dim=1)
        
        # Replace the module
        setattr(parent_module, module_name, new_module)
        
        print(f"Replaced {module_name} with RowParallelLinear")
    
    def _load_sharded_state_dict(self, new_module, original_module, shard_dim):
        """
        Load state dict with proper weight sharding.
        
        Args:
            new_module: The new TP module
            original_module: The original nn.Linear module  
            shard_dim: Dimension to shard (0 for column, 1 for row)
        """
        original_state = original_module.state_dict()
        tp_rank = self.parallel_context.get_rank_in("tp")
        tp_size = self.parallel_context.parallel_dims["tp"]
        
        # Shard the weight
        original_weight = original_state['weight']
        shard_size = original_weight.size(shard_dim) // tp_size
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size
        
        if shard_dim == 0:  # Column parallel - shard output dimension
            sharded_weight = original_weight[start_idx:end_idx, :]
        else:  # Row parallel - shard input dimension  
            sharded_weight = original_weight[:, start_idx:end_idx]
        
        # Create new state dict
        new_state = {'weight': sharded_weight}
        
        # Handle bias
        if 'bias' in original_state and original_state['bias'] is not None:
            original_bias = original_state['bias']
            if shard_dim == 0:  # Column parallel - shard bias
                sharded_bias = original_bias[start_idx:end_idx]
            else:  # Row parallel - keep full bias (will be reduced)
                sharded_bias = original_bias
            new_state['bias'] = sharded_bias
        
        # Load the sharded state dict
        new_module.load_state_dict(new_state)
    
    def get_tp_info(self):
        """Get information about the TP configuration."""
        return {
            'tp_size': self.parallel_context.parallel_dims["tp"],
            'tp_rank': self.parallel_context.get_rank_in("tp"),
            'column_parallel_modules': self.column_linear_names,
            'row_parallel_modules': self.row_linear_names,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


# Compatibility alias for backward compatibility
TP = TPWrapper


def apply_tensor_parallel(model: nn.Module,
                         parallel_context: ParallelContext,
                         column_linear_names: Optional[List[str]] = None,
                         row_linear_names: Optional[List[str]] = None,
                         auto_tune: bool = False) -> TPWrapper:
    """
    Apply tensor parallelism to a model.
    
    Args:
        model (nn.Module): The model to apply TP to
        parallel_context (ParallelContext): The parallel context
        column_linear_names (List[str], optional): Names of modules for column parallel
        row_linear_names (List[str], optional): Names of modules for row parallel
        auto_tune (bool): Whether to enable auto tuning
        
    Returns:
        TPWrapper: The wrapped model with TP applied
        
    Example:
        >>> # Initialize distributed and create context
        >>> dist.init_process_group(backend="nccl", init_method="env://")
        >>> ctx = ParallelContext({"tp": 2})
        >>> 
        >>> # Apply TP to model
        >>> model = MyTransformerModel()
        >>> tp_model = apply_tensor_parallel(
        ...     model=model,
        ...     parallel_context=ctx,
        ...     column_linear_names=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        ...     row_linear_names=["o_proj", "down_proj"]
        ... )
        >>> 
        >>> # Use the TP model
        >>> output = tp_model(input_ids)
    """
    return TPWrapper(
        model=model,
        parallel_context=parallel_context,
        column_linear_names=column_linear_names,
        row_linear_names=row_linear_names,
        auto_tune=auto_tune
    )

