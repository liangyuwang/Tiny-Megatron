# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn

from .module import (
    Linear,
    LayerNorm,
    Embedding,
)
from ..utils.wrapper import target_modules, get_init_args
from ..utils.comm import ParallelContext

class DPWrapper(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 parallel_context: ParallelContext,
                 auto_tune: bool = False):
        """
        Initialize the DP wrapper.
        
        Args:
            model (nn.Module): The original model to wrap
            parallel_context (ParallelContext): The parallel context for DP communication
            auto_tune (bool): Whether to enable auto tuning for optimized kernels
        """
        super().__init__()
        self.auto_tune = auto_tune
        self._wrap_layers(model)
        self._error_handling(model)
        self.module = model
        self.parallel_context = parallel_context
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        if self.require_backward_grad_sync:
            self.enable_grad_sync()
        self.require_backward_grad_sync = False
        return self.module(*args, **kwargs)

    def enable_grad_sync(self):
        for param in self.module.parameters():
            setattr(param, 'bwd_sync', True)

    def _wrap_layers(self, model):
        """Wrap layers with DP-enabled modules."""
        def _replace_module_recursive(model, path=''):
            for child_name, child in model.named_children():
                full_name = f"{path}.{child_name}" if path else child_name
                if isinstance(child, tuple(target_modules)):
                    module_class = _supported_modules[type(child)]
                    child_init_args = get_init_args(child)
                    new_module = module_class(**child_init_args, auto_tune=self.auto_tune)
                    if hasattr(child, 'parameters') and len(list(child.parameters())) > 0:
                        child_device = next(child.parameters()).device
                        new_module = new_module.to(child_device)
                    
                    # Load state dict with strict=False to handle bias differences
                    new_module.load_state_dict(child.state_dict(), strict=False)
                    new_module.train(child.training)
                    setattr(model, child_name, new_module)
                else:
                    _replace_module_recursive(child, full_name)
        _replace_module_recursive(model)

    def _error_handling(self, model):
        """Ensure all parameters have bwd_sync attribute."""
        for name, param in model.named_parameters():
            if not hasattr(param, "bwd_sync"):
                raise NotImplementedError(f"Module {name} is not supported yet. Currently support modules: [nn.Linear, nn.LayerNorm, nn.Embedding].")


_supported_modules = {
    nn.Linear: Linear,
    nn.LayerNorm: LayerNorm,
    nn.Embedding: Embedding,
}


def apply_data_parallel(model: nn.Module,
                       parallel_context: ParallelContext,
                       auto_tune: bool = False) -> DPWrapper:
    """
    Apply data parallelism to a model.
    
    Args:
        model (nn.Module): The model to apply DP to
        parallel_context (ParallelContext): The parallel context
        auto_tune (bool): Whether to enable auto tuning for optimized kernels
        
    Returns:
        DPWrapper: The wrapped model with DP applied
        
    Example:
        >>> # Initialize distributed and create context
        >>> dist.init_process_group(backend="nccl", init_method="env://")
        >>> ctx = ParallelContext({"dp": world_size})
        >>> 
        >>> # Apply DP to model
        >>> model = MyTransformerModel()
        >>> dp_model = apply_data_parallel(
        ...     model=model,
        ...     parallel_context=ctx,
        ...     auto_tune=True
        ... )
        >>> 
        >>> # Use the DP model
        >>> dp_model.require_backward_grad_sync = True
        >>> output = dp_model(input_ids)
    """
    return DPWrapper(
        model=model,
        parallel_context=parallel_context,
        auto_tune=auto_tune
    )


# Compatibility alias for backward compatibility
DDP = DPWrapper
