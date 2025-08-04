# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any

from ..utils.comm import ParallelContext
from ...module import Linear, LayerNorm, Embedding


class PPCommunicationMixin:
    """
    Mixin class that provides P2P communication functionality for PP modules.
    """
    
    def __init__(self, parallel_context: ParallelContext, pp_rank: int):
        """
        Initialize PP communication.
        
        Args:
            parallel_context: Parallel context for rank mapping
            pp_rank: Current pipeline parallel rank
        """
        self.parallel_context = parallel_context
        self.pp_rank = pp_rank
        self.pp_size = parallel_context.parallel_dims["pp"]
        self.pp_group = parallel_context.get_group("pp")
        
        # Calculate previous and next ranks in pipeline
        self.prev_rank = self._pp_rank_to_global_rank(pp_rank - 1) if pp_rank > 0 else None
        self.next_rank = self._pp_rank_to_global_rank(pp_rank + 1) if pp_rank < self.pp_size - 1 else None
        
        # Determine communication pattern
        self.is_first_stage = pp_rank == 0
        self.is_last_stage = pp_rank == self.pp_size - 1
    
    def is_pipeline_first_stage(self) -> bool:
        """Check if this is the first pipeline stage."""
        return self.is_first_stage
    
    def is_pipeline_last_stage(self) -> bool:
        """Check if this is the last pipeline stage."""
        return self.is_last_stage
    
    def _pp_rank_to_global_rank(self, pp_rank: int) -> int:
        """Convert PP rank to global rank."""
        coords = self.parallel_context.get_coord_dict()
        target_coords = coords.copy()
        target_coords['pp'] = pp_rank
        
        return self.parallel_context._coords_to_rank(
            [target_coords[name] for name in self.parallel_context.dim_names]
        )
    
    def _recv_from_prev_stage(self, input_shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Receive activation from the previous stage.
        
        Args:
            input_shape: Expected shape of input tensor
            dtype: Expected data type
            device: Target device
            
        Returns:
            Received tensor from previous stage
        """
        if self.is_pipeline_first_stage():
            raise RuntimeError("First stage should not receive from previous stage")
        
        # Create empty tensor to receive data
        input_tensor = torch.empty(input_shape, dtype=dtype, device=device)
        
        # Receive from previous rank
        dist.recv(tensor=input_tensor, src=self.prev_rank, group=self.pp_group)
        
        return input_tensor
    
    def _send_to_next_stage(self, output_tensor: torch.Tensor) -> None:
        """
        Send activation to the next stage.
        
        Args:
            output_tensor: Tensor to send to next stage
        """
        if self.is_pipeline_last_stage():
            return  # Last stage doesn't send to next
        
        # Send to next rank
        dist.send(tensor=output_tensor.contiguous(), dst=self.next_rank, group=self.pp_group)


class PPLinear(Linear, PPCommunicationMixin):
    """
    Pipeline Parallel Linear layer (simplified - no module-level communication).
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 device=None, 
                 dtype=None, 
                 auto_tune: bool = False,
                 parallel_context: Optional[ParallelContext] = None,
                 pp_rank: Optional[int] = None):
        """Initialize PP Linear layer."""
        Linear.__init__(self, in_features, out_features, bias, device, dtype, auto_tune)
        
        if parallel_context is not None and pp_rank is not None:
            PPCommunicationMixin.__init__(self, parallel_context, pp_rank)
            self.pp_enabled = True
        else:
            self.pp_enabled = False
    

class PPLayerNorm(LayerNorm, PPCommunicationMixin):
    """
    Pipeline Parallel LayerNorm (simplified - no module-level communication).
    """
    
    def __init__(self, 
                 normalized_shape, 
                 eps: float = 1e-5, 
                 elementwise_affine: bool = True, 
                 bias: bool = True,  # Add bias parameter for Tiny-Megatron LayerNorm
                 device=None, 
                 dtype=None, 
                 auto_tune: bool = False,
                 parallel_context: Optional[ParallelContext] = None,
                 pp_rank: Optional[int] = None):
        """Initialize PP LayerNorm."""
        LayerNorm.__init__(self, normalized_shape, eps, elementwise_affine, bias, device, dtype, auto_tune)
        
        if parallel_context is not None and pp_rank is not None:
            PPCommunicationMixin.__init__(self, parallel_context, pp_rank)
            self.pp_enabled = True
        else:
            self.pp_enabled = False
    

class PPEmbedding(Embedding, PPCommunicationMixin):
    """
    Pipeline Parallel Embedding (simplified - no module-level communication).
    """
    
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 padding_idx: Optional[int] = None, 
                 max_norm: Optional[float] = None, 
                 norm_type: float = 2.0, 
                 scale_grad_by_freq: bool = False, 
                 sparse: bool = False, 
                 device=None, 
                 dtype=None, 
                 auto_tune: bool = False,
                 parallel_context: Optional[ParallelContext] = None,
                 pp_rank: Optional[int] = None):
        """Initialize PP Embedding."""
        Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx, max_norm, 
                          norm_type, scale_grad_by_freq, sparse, device, dtype, auto_tune)
        
        if parallel_context is not None and pp_rank is not None:
            PPCommunicationMixin.__init__(self, parallel_context, pp_rank)
            self.pp_enabled = True
        else:
            self.pp_enabled = False
    

# Module mapping for PP wrapper
PP_MODULE_MAPPING = {
    Linear: PPLinear,
    LayerNorm: PPLayerNorm,
    Embedding: PPEmbedding,
    nn.Linear: PPLinear,
    nn.LayerNorm: PPLayerNorm,
    nn.Embedding: PPEmbedding,
} 