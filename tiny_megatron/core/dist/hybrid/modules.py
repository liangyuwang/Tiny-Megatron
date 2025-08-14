# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.distributed as dist
from typing import Optional

from ...module import Linear, LayerNorm, Embedding
from ...module.ops import linear, layernorm, embedding
from ..utils.comm import ParallelContext


class HybridLinear(Linear):
    """
    Hybrid Linear module supporting Tensor Parallelism (TP) + Pipeline Parallelism (PP) + Data Parallelism (DP).
    
    Inherits from the base Linear module and overrides forward_callback and backward_callback
    to insert hybrid parallelism communication logic.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 auto_tune: bool = False,
                 parallel_context: Optional[ParallelContext] = None,
                 tp_mode: str = "none",  # "none", "column", "row"
                 pp_enabled: bool = False,
                 dp_enabled: bool = False):
        """
        Initialize Hybrid Linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features  
            bias: Whether to use bias
            device: Target device
            dtype: Target dtype
            auto_tune: Whether to enable auto tuning
            parallel_context: Context for distributed parallelism
            tp_mode: Tensor parallelism mode ("none", "column", "row")
            pp_enabled: Whether pipeline parallelism is enabled
            dp_enabled: Whether data parallelism is enabled
        """
        self.parallel_context = parallel_context
        self.tp_mode = tp_mode
        self.pp_enabled = pp_enabled
        self.dp_enabled = dp_enabled
        
        # Get parallel sizes and ranks
        if parallel_context:
            self.tp_size = parallel_context.parallel_dims.get("tp", 1)
            self.pp_size = parallel_context.parallel_dims.get("pp", 1) 
            self.dp_size = parallel_context.parallel_dims.get("dp", 1)
            self.tp_rank = parallel_context.get_rank_in("tp") if self.tp_size > 1 else 0
            self.pp_rank = parallel_context.get_rank_in("pp") if self.pp_size > 1 else 0
            self.dp_rank = parallel_context.get_rank_in("dp") if self.dp_size > 1 else 0
        else:
            self.tp_size = self.pp_size = self.dp_size = 1
            self.tp_rank = self.pp_rank = self.dp_rank = 0
        
        # Store original dimensions for reference
        self.original_in_features = in_features
        self.original_out_features = out_features
        
        # Calculate actual dimensions based on TP mode
        actual_in_features, actual_out_features = self._calculate_dimensions(in_features, out_features)
        
        # For row parallel, only the first TP rank has bias to avoid duplication
        if bias and tp_mode == "row" and self.tp_rank != 0:
            bias = False
        
        # Initialize base Linear with potentially modified dimensions
        super().__init__(actual_in_features, actual_out_features, bias, device, dtype, auto_tune)
        
        # Setup pipeline communication if enabled
        if self.pp_enabled and self.pp_size > 1:
            self._setup_pp_communication()
        else:
            # Initialize PP attributes to None when PP is not enabled
            self.pp_group = None
            self.prev_rank = None
            self.next_rank = None
    
    def _calculate_dimensions(self, in_features: int, out_features: int):
        """Calculate actual tensor dimensions based on TP mode."""
        if self.tp_mode == "column":
            # Column parallel: split output dimension
            local_out_features = out_features // self.tp_size
            return in_features, local_out_features
        elif self.tp_mode == "row":
            # Row parallel: split input dimension
            local_in_features = in_features // self.tp_size
            return local_in_features, out_features
        else:
            # No TP
            return in_features, out_features
    
    def _setup_pp_communication(self):
        """Setup pipeline parallelism communication parameters."""
        if not self.parallel_context:
            return
            
        # Get PP communication group and ranks
        self.pp_group = self.parallel_context.get_group("pp")
        
        # Calculate previous and next ranks in pipeline
        if self.pp_rank > 0:
            prev_coord = self.parallel_context.get_coord_dict()
            prev_coord["pp"] = self.pp_rank - 1
            self.prev_rank = self.parallel_context._coords_to_rank(
                [prev_coord[name] for name in self.parallel_context.dim_names]
            )
        else:
            self.prev_rank = None
            
        if self.pp_rank < self.pp_size - 1:
            next_coord = self.parallel_context.get_coord_dict()
            next_coord["pp"] = self.pp_rank + 1
            self.next_rank = self.parallel_context._coords_to_rank(
                [next_coord[name] for name in self.parallel_context.dim_names]
            )
        else:
            self.next_rank = None
    
    def forward_callback(self, ctx, input, weight, bias, runtime_tuner):
        """
        Override forward callback to insert hybrid parallelism communication.
        """
        # Handle pipeline parallelism input reception
        if self.pp_enabled and self.pp_rank > 0 and self.prev_rank is not None:
            received_input = torch.empty_like(input)
            dist.recv(tensor=received_input, src=self.prev_rank, group=self.pp_group)
            input = received_input
        
        # Save tensors for backward
        ctx.save_for_backward(input, weight, bias)
        
        # Perform the actual linear computation
        output = linear.linear_forward(input, weight, bias, runtime_tuner)
        
        # Handle tensor parallelism communication in forward pass
        if self.tp_mode == "row" and self.tp_size > 1:
            # Row parallel: all_reduce output across TP ranks
            tp_group = self.parallel_context.get_group("tp")
            dist.all_reduce(output, group=tp_group)
        
        # Handle pipeline parallelism output transmission
        if self.pp_enabled and self.pp_rank < self.pp_size - 1 and self.next_rank is not None:
            dist.send(tensor=output, dst=self.next_rank, group=self.pp_group)
        
        return ctx, output
    
    def backward_callback(self, ctx, grad_output, runtime_tuner):
        """
        Override backward callback to insert hybrid parallelism communication.
        TODO: The scheduling order of grad_input and grad_weight determines which type of overlap is favored.
              For example, if grad_input is computed first, then grad_weight is computed,
              then the overlap is between grad_input all reduce and grad_weight compute. (TP)
              If grad_weight is computed first, then grad_input is computed,
              then the overlap is between grad_weight all reduce and grad_input compute. (DP)
              We should add a flag to the runtime_tuner to control the scheduling order.
        """
        input, weight, bias = ctx.saved_tensors
        
        # Handle pipeline parallelism gradient reception
        if self.pp_enabled and self.pp_rank < self.pp_size - 1 and self.next_rank is not None:
            received_grad = torch.empty_like(grad_output)
            dist.recv(tensor=received_grad, src=self.next_rank, group=self.pp_group)
            grad_output = received_grad
        
        # Initialize communication handles
        tp_handle = None
        
        # Compute gradients
        if ctx.needs_input_grad[0]:
            grad_input = linear.linear_input_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_input = None

        # Handle tensor parallelism communication in backward pass
        if self.tp_mode == "column" and self.tp_size > 1 and grad_input is not None:
            # Column parallel: all_reduce grad_input across TP ranks
            tp_group = self.parallel_context.get_group("tp")
            tp_handle = dist.all_reduce(grad_input, group=tp_group, async_op=True)
        
        if ctx.needs_input_grad[1]:
            grad_weight = linear.linear_weight_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = linear.linear_bias_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_bias = None
        
        # Wait for tensor parallelism communication to complete
        if tp_handle is not None:
            tp_handle.wait()

        # Handle pipeline parallelism gradient transmission
        if self.pp_enabled and self.pp_rank > 0 and self.prev_rank is not None and grad_input is not None:
            dist.send(tensor=grad_input, dst=self.prev_rank, group=self.pp_group)
        
        # Handle data parallelism gradient synchronization
        if self.dp_enabled and self.dp_size > 1:
            dp_group = self.parallel_context.get_group("dp")
            if grad_weight is not None:
                dist.all_reduce(grad_weight, group=dp_group)
                grad_weight = grad_weight / self.dp_size
            if grad_bias is not None:
                dist.all_reduce(grad_bias, group=dp_group)
                grad_bias = grad_bias / self.dp_size
        
        # Validate gradient shapes
        if grad_input is not None and grad_input.shape != input.shape:
            raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
        if grad_bias is not None and grad_bias.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")
        
        return grad_input, grad_weight, grad_bias


class HybridLayerNorm(LayerNorm):
    """
    Hybrid LayerNorm module supporting Pipeline Parallelism.
    Note: LayerNorm doesn't need TP since it's not a linear transformation.
    """
    
    def __init__(self,
                 normalized_shape,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 auto_tune: bool = False,
                 parallel_context: Optional[ParallelContext] = None,
                 pp_enabled: bool = False):
        """
        Initialize Hybrid LayerNorm.
        
        Args:
            normalized_shape: Input shape from an expected input
            eps: Value added to denominator for numerical stability
            elementwise_affine: Whether to use learnable affine parameters
            bias: Whether to use bias
            device: Target device
            dtype: Target dtype
            auto_tune: Whether to enable auto tuning
            parallel_context: Context for distributed parallelism  
            pp_enabled: Whether pipeline parallelism is enabled
        """
        self.parallel_context = parallel_context
        self.pp_enabled = pp_enabled
        
        # Initialize base LayerNorm
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype, auto_tune)
        
        # Setup pipeline communication if enabled
        if self.pp_enabled and parallel_context:
            self.pp_size = parallel_context.parallel_dims.get("pp", 1)
            self.pp_rank = parallel_context.get_rank_in("pp") if self.pp_size > 1 else 0
            if self.pp_size > 1:
                self._setup_pp_communication()
            else:
                # Initialize PP attributes to None when PP is not enabled
                self.pp_group = None
                self.prev_rank = None
                self.next_rank = None
        else:
            # Initialize PP attributes to None when PP is not enabled
            self.pp_group = None
            self.prev_rank = None
            self.next_rank = None
    
    def _setup_pp_communication(self):
        """Setup pipeline parallelism communication parameters."""
        self.pp_group = self.parallel_context.get_group("pp")
        
        if self.pp_rank > 0:
            prev_coord = self.parallel_context.get_coord_dict()
            prev_coord["pp"] = self.pp_rank - 1
            self.prev_rank = self.parallel_context._coords_to_rank(
                [prev_coord[name] for name in self.parallel_context.dim_names]
            )
        else:
            self.prev_rank = None
            
        if self.pp_rank < self.pp_size - 1:
            next_coord = self.parallel_context.get_coord_dict()
            next_coord["pp"] = self.pp_rank + 1
            self.next_rank = self.parallel_context._coords_to_rank(
                [next_coord[name] for name in self.parallel_context.dim_names]
            )
        else:
            self.next_rank = None
    
    def forward_callback(self, ctx, input, weight, bias, eps, runtime_tuner):
        """
        Override forward callback to insert pipeline parallelism communication.
        """
        # Handle pipeline parallelism input reception
        if self.pp_enabled and self.pp_rank > 0 and self.prev_rank is not None:
            received_input = torch.empty_like(input)
            dist.recv(tensor=received_input, src=self.prev_rank, group=self.pp_group)
            input = received_input
        
        # Perform LayerNorm computation
        output, mean, rstd, args = layernorm.layernorm_fwd(input, weight, bias, eps, runtime_tuner)
        ctx.save_for_backward(input, weight, bias, mean, rstd)
        ctx.args = args
        
        # Handle pipeline parallelism output transmission  
        if self.pp_enabled and self.pp_rank < self.pp_size - 1 and self.next_rank is not None:
            dist.send(tensor=output, dst=self.next_rank, group=self.pp_group)
        
        return ctx, output
    
    def backward_callback(self, ctx, grad_output, eps, runtime_tuner):
        """
        Override backward callback to insert pipeline parallelism communication.
        """
        # Handle pipeline parallelism gradient reception
        if self.pp_enabled and self.pp_rank < self.pp_size - 1 and self.next_rank is not None:
            received_grad = torch.empty_like(grad_output)
            dist.recv(tensor=received_grad, src=self.next_rank, group=self.pp_group)
            grad_output = received_grad
        
        # Perform LayerNorm backward computation
        input, weight, bias, mean, rstd = ctx.saved_tensors
        args = {
            'BLOCK_SIZE': ctx.args['BLOCK_SIZE'],
            'num_warps': ctx.args['num_warps'],
            'eps': eps,
        }
        dx, dw_, db_, args = layernorm.layernorm_dx(grad_output, input, weight, bias, mean, rstd, args, runtime_tuner)
        dw, db = layernorm.layernorm_dwdb(weight, bias, dw_, db_, args, runtime_tuner)
        
        # Handle pipeline parallelism gradient transmission
        if self.pp_enabled and self.pp_rank > 0 and self.prev_rank is not None and dx is not None:
            dist.send(tensor=dx, dst=self.prev_rank, group=self.pp_group)
        
        # Validate gradient shapes
        if dx is not None and dx.shape != input.shape:
            raise RuntimeError(f"grad_input shape {dx.shape} is not equal to input shape {input.shape}")
        if dw is not None and dw.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {dw.shape} is not equal to weight shape {weight.shape}")
        if db is not None and db.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {db.shape} is not equal to bias shape {bias.shape}")

        return dx, dw, db


class HybridEmbedding(Embedding):
    """
    Hybrid Embedding module supporting Tensor Parallelism and Pipeline Parallelism.
    """
    
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _weight: Optional[torch.Tensor] = None,
                 _freeze: bool = False,
                 device=None,
                 dtype=None,
                 auto_tune: bool = False,
                 parallel_context: Optional[ParallelContext] = None,
                 tp_enabled: bool = False,
                 pp_enabled: bool = False):
        """
        Initialize Hybrid Embedding.
        
        Args:
            num_embeddings: Size of the dictionary of embeddings
            embedding_dim: Size of each embedding vector
            padding_idx: If given, pads the output with the embedding vector at padding_idx
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm
            norm_type: The p of the p-norm to compute for the max_norm option. Default 2
            scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of the words in the mini-batch
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor
            _weight: If specified, use this Tensor as the weight
            _freeze: If True, the tensor does not get updated in the learning process
            device: Target device
            dtype: Target dtype
            auto_tune: Whether to enable auto tuning
            parallel_context: Context for distributed parallelism
            tp_enabled: Whether tensor parallelism is enabled
            pp_enabled: Whether pipeline parallelism is enabled
        """
        self.parallel_context = parallel_context
        self.tp_enabled = tp_enabled
        self.pp_enabled = pp_enabled
        
        # Get parallel configuration
        if parallel_context:
            self.tp_size = parallel_context.parallel_dims.get("tp", 1)
            self.pp_size = parallel_context.parallel_dims.get("pp", 1)
            self.tp_rank = parallel_context.get_rank_in("tp") if self.tp_size > 1 else 0
            self.pp_rank = parallel_context.get_rank_in("pp") if self.pp_size > 1 else 0
        else:
            self.tp_size = self.pp_size = 1
            self.tp_rank = self.pp_rank = 0
        
        # Store original embedding dimension
        self.original_embedding_dim = embedding_dim
        
        # Calculate embedding dimensions for TP
        if self.tp_enabled and self.tp_size > 1:
            # Shard embedding dimension across TP ranks
            local_embedding_dim = embedding_dim // self.tp_size
        else:
            local_embedding_dim = embedding_dim
        
        # Initialize base Embedding with potentially modified dimensions
        super().__init__(num_embeddings, local_embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype, auto_tune)
        
        # Setup pipeline communication if enabled
        if self.pp_enabled and self.pp_size > 1:
            self._setup_pp_communication()
        else:
            # Initialize PP attributes to None when PP is not enabled
            self.pp_group = None
            self.prev_rank = None
            self.next_rank = None
    
    def _setup_pp_communication(self):
        """Setup pipeline parallelism communication parameters."""
        self.pp_group = self.parallel_context.get_group("pp")
        
        if self.pp_rank < self.pp_size - 1:
            next_coord = self.parallel_context.get_coord_dict()
            next_coord["pp"] = self.pp_rank + 1
            self.next_rank = self.parallel_context._coords_to_rank(
                [next_coord[name] for name in self.parallel_context.dim_names]
            )
        else:
            self.next_rank = None
    
    def forward_callback(self, ctx, input, weight, padding_idx, max_norm, norm_type, runtime_tuner):
        """
        Override forward callback to insert hybrid parallelism communication.
        """
        ctx.save_for_backward(input, weight)
        
        # Perform embedding lookup
        output = embedding.embedding_forward(input, weight, padding_idx, max_norm, norm_type, runtime_tuner)
        
        # Handle tensor parallelism - all_gather to reconstruct full embedding
        if self.tp_enabled and self.tp_size > 1:
            tp_group = self.parallel_context.get_group("tp")
            output_list = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(output_list, output, group=tp_group)
            output = torch.cat(output_list, dim=-1)
        
        # Handle pipeline parallelism output transmission
        if self.pp_enabled and self.next_rank is not None:
            dist.send(tensor=output, dst=self.next_rank, group=self.pp_group)
        
        return ctx, output
    
    def backward_callback(self, ctx, grad_output, padding_idx, max_norm, norm_type, runtime_tuner):
        """
        Override backward callback to insert hybrid parallelism communication.
        """
        input, weight = ctx.saved_tensors
        
        # Handle tensor parallelism gradient slicing
        if self.tp_enabled and self.tp_size > 1:
            # Slice grad_output to get local portion
            local_embedding_dim = self.original_embedding_dim // self.tp_size
            start_idx = self.tp_rank * local_embedding_dim
            end_idx = start_idx + local_embedding_dim
            grad_output = grad_output[..., start_idx:end_idx]
        
        # Compute weight gradient
        if ctx.needs_input_grad[1]:
            grad_weight = embedding.embedding_weight_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_weight = None

        # Validate gradient shapes
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")

        return grad_weight 