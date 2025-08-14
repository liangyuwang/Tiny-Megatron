# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any

from ..utils.comm import ParallelContext
from .modules import PP_MODULE_MAPPING, PPCommunicationMixin


class PPWrapper(nn.Module):
    """
    Pipeline Parallel Wrapper that replaces modules with PP-enabled versions.
    
    This approach keeps the model structure intact while enabling P2P communication
    at the module level. Only transformer blocks are distributed across PP ranks.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 parallel_context: ParallelContext,
                 block_names: List[str]):
        """
        Initialize the PP wrapper.
        
        Args:
            model (nn.Module): The original model to wrap
            parallel_context (ParallelContext): The parallel context with PP configuration
            block_names (List[str]): List of block container names (e.g., ["transformer.h"])
        """
        super().__init__()
        
        if parallel_context is None:
            raise ValueError("parallel_context must be provided for PP wrapper")
        
        self.parallel_context = parallel_context
        self.pp_rank = parallel_context.get_rank_in("pp")
        self.pp_size = parallel_context.parallel_dims["pp"]
        self.block_names = block_names
        
        # Calculate previous and next global ranks for communication
        self.prev_rank = self._pp_rank_to_global_rank(self.pp_rank - 1) if self.pp_rank > 0 else None
        self.next_rank = self._pp_rank_to_global_rank(self.pp_rank + 1) if self.pp_rank < self.pp_size - 1 else None
        
        # Set target device for PP parameters based on rank
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{parallel_context.rank}")
        else:
            self.device = torch.device("cpu")
        
        # Wrap the model by replacing modules
        self.model = self._wrap_model(model)
        
        # Move model to target device
        self.model = self.model.to(self.device)
        
        print(f"[Rank {parallel_context.rank}] PPWrapper initialized:")
        print(f"  - PP rank: {self.pp_rank}/{self.pp_size}")
        print(f"  - Coordinates: {parallel_context.get_coord_dict()}")
    
    def _pp_rank_to_global_rank(self, pp_rank: int) -> int:
        """Convert PP rank to global rank."""
        coords = self.parallel_context.get_coord_dict()
        target_coords = coords.copy()
        target_coords['pp'] = pp_rank
        
        return self.parallel_context._coords_to_rank(
            [target_coords[name] for name in self.parallel_context.dim_names]
        )
    
    def forward(self, *args, **kwargs):
        """Forward pass through the pipeline parallel model with PP orchestration."""
        return self._pp_forward(*args, **kwargs)
    
    def _pp_forward(self, *args, **kwargs):
        """
        Pipeline parallel forward pass with stage-level communication.
        
        Each stage processes ONLY its assigned layers, then communicates once.
        """
        # Stage 0 (first): process embedding + assigned blocks, send to next
        if self.pp_rank == 0:
            # Process only the first stage components
            hidden_states = self._process_first_stage(*args, **kwargs)
            
            # Send to next stage if not single stage
            if self.pp_size > 1:
                self._send_to_next_stage(hidden_states)
            else:
                # Single stage - process everything
                return self._process_complete_model(*args, **kwargs)
            
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
    
    def _send_to_next_stage(self, tensor: torch.Tensor) -> None:
        """Send tensor to next PP stage."""
        if self.next_rank is not None:
            pp_group = self.parallel_context.get_group("pp")
            dist.send(tensor.contiguous(), dst=self.next_rank, group=pp_group)
    
    def _recv_from_prev_stage(self) -> torch.Tensor:
        """Receive tensor from previous PP stage."""
        if self.prev_rank is None:
            raise RuntimeError("Cannot receive from previous stage: prev_rank is None")
            
        pp_group = self.parallel_context.get_group("pp")
        
        # Create tensor with appropriate shape (hardcoded for now)
        # In practice, this would be negotiated or pre-agreed
        batch_size, seq_len, hidden_size = 1, 1024, 768
        tensor = torch.empty(batch_size, seq_len, hidden_size, 
                           dtype=torch.float32, device=self.device)
        
        dist.recv(tensor, src=self.prev_rank, group=pp_group)
        return tensor
    
    def _process_middle_stage(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process through middle stage layers."""
        # For middle stages, we need to process the received hidden states
        # through our transformer blocks
        x = hidden_states
        for name, module in self.model.named_children():
            if name == 'transformer':
                for block_name, block in module.h.named_children():
                    x = block(x)
            # Skip other modules (embedding, final layers)
        return x
    
    def _process_last_stage(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Process through last stage layers and compute loss if needed."""
        # Process through our transformer blocks
        x = hidden_states
        for name, module in self.model.named_children():
            if name == 'transformer':
                # Process through our blocks
                for block_name, block in module.h.named_children():
                    x = block(x)
                # Apply final layer norm
                x = module.ln_f(x)
            elif name == 'lm_head':
                # Apply language model head
                logits = module(x)
                
                # Compute loss if targets provided
                loss = None
                if len(args) > 1:
                    targets = args[1]
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                
                return logits, loss
        
        return x
    
    def _pp_forward_first_rank(self, *args, **kwargs):
        """Forward pass for first PP rank."""
        # Process input through our layers
        # PP modules will automatically send outputs to next stage
        output = self.model(*args, **kwargs)
        return output
    
    def _pp_forward_last_rank(self, *args, **kwargs):
        """Forward pass for last PP rank."""
        # Process through our layers
        # PP modules will automatically receive inputs from previous stage
        output = self.model(*args, **kwargs)
        return output
    
    def _pp_forward_middle_rank(self, *args, **kwargs):
        """Forward pass for middle PP ranks."""
        # Process through our layers
        # PP modules will automatically receive from previous and send to next stage
        output = self.model(*args, **kwargs)
        return output
    
    def _has_embedding_layer(self) -> bool:
        """Check if this stage contains embedding layers."""
        for name, module in self.model.named_modules():
            if 'wte' in name or 'embedding' in name.lower():
                return True
        return False

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Replace modules with PP-enabled versions.
        
        Args:
            model: Original model
            
        Returns:
            Model with PP modules
        """
        # Create a copy of the model
        import copy
        model_copy = copy.deepcopy(model)
        
        # Process each block container
        for block_name in self.block_names:
            self._process_block_container(model_copy, block_name)
        
        return model_copy
    
    def _process_block_container(self, model: nn.Module, block_name: str):
        """
        Process blocks in a container - replace modules in owned blocks, replace others with Identity.
        
        Args:
            model: Model to modify
            block_name: Container name (e.g., "transformer.h")
        """
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
            
            print(f"[Rank {self.parallel_context.rank}] PP blocks: {start_idx}-{end_idx-1}")
            
            # Process each block
            for i in range(total_blocks):
                if start_idx <= i < end_idx:
                    # This block belongs to current rank - replace modules with PP versions
                    self._replace_modules_in_block(block_container[i], f"{block_name}.{i}")
                else:
                    # This block belongs to other ranks - replace with identity
                    block_container[i] = nn.Identity()
    
    def _replace_modules_in_block(self, block: nn.Module, block_path: str):
        """
        Replace modules in a block with PP-enabled versions.
        
        Args:
            block: Block to process
            block_path: Path to the block (for debugging)
        """
        modules_replaced = 0
        
        for name, module in list(block.named_children()):
            module_type = type(module)
            
            if module_type in PP_MODULE_MAPPING:
                # Replace with PP version
                pp_module_class = PP_MODULE_MAPPING[module_type]
                
                # Get the original module's arguments
                pp_module = self._create_pp_module(module, pp_module_class)
                
                # Replace the module
                setattr(block, name, pp_module)
                modules_replaced += 1
            
            # Recursively process child modules
            if len(list(module.children())) > 0:
                self._replace_modules_in_block(module, f"{block_path}.{name}")
        
        if modules_replaced > 0:
            print(f"[Rank {self.parallel_context.rank}] Replaced {modules_replaced} modules in {block_path}")
    
    def _create_pp_module(self, original_module: nn.Module, pp_module_class) -> nn.Module:
        """
        Create a PP version of the original module.
        
        Args:
            original_module: Original module
            pp_module_class: PP module class
            
        Returns:
            PP module instance
        """
        # Get original module parameters
        state_dict = original_module.state_dict()
        
        # Create PP module with same parameters
        if hasattr(original_module, 'in_features') and hasattr(original_module, 'out_features'):
            # Linear layer
            pp_module = pp_module_class(
                in_features=original_module.in_features,
                out_features=original_module.out_features,
                bias=original_module.bias is not None,
                parallel_context=self.parallel_context,
                pp_rank=self.pp_rank
            )
        elif hasattr(original_module, 'normalized_shape'):
            # LayerNorm
            pp_module = pp_module_class(
                normalized_shape=original_module.normalized_shape,
                eps=original_module.eps,
                elementwise_affine=original_module.elementwise_affine,
                bias=getattr(original_module, 'use_bias', True),  # Use bias from original or default True
                parallel_context=self.parallel_context,
                pp_rank=self.pp_rank
            )
        elif hasattr(original_module, 'num_embeddings') and hasattr(original_module, 'embedding_dim'):
            # Embedding
            pp_module = pp_module_class(
                num_embeddings=original_module.num_embeddings,
                embedding_dim=original_module.embedding_dim,
                padding_idx=original_module.padding_idx,
                parallel_context=self.parallel_context,
                pp_rank=self.pp_rank
            )
        else:
            # Generic fallback - try to copy basic parameters
            pp_module = pp_module_class(
                parallel_context=self.parallel_context,
                pp_rank=self.pp_rank
            )
        
        # Load the state dict
        pp_module.load_state_dict(state_dict, strict=False)
        
        return pp_module
    
    def get_pp_info(self):
        """Get information about the PP configuration."""
        return {
            'pp_size': self.pp_size,
            'pp_rank': self.pp_rank,
            'block_names': self.block_names,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def _process_first_stage(self, *args, **kwargs) -> torch.Tensor:
        """Process first stage: embedding + assigned transformer blocks."""
        # Get input
        idx = args[0]
        
        # Process embedding
        x = None
        for name, module in self.model.named_children():
            if name == 'transformer':
                # Process token and position embeddings
                B, T = idx.size()
                pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
                pos_emb = module.wpe(pos)
                tok_emb = module.wte(idx)
                x = tok_emb + pos_emb
                
                # Process only assigned transformer blocks
                for block_name, block in module.h.named_children():
                    x = block(x)
                break
        
        return x
    
    def _process_complete_model(self, *args, **kwargs):
        """Process complete model for single-stage PP."""
        return self.model(*args, **kwargs)


def apply_pipeline_parallel(model: nn.Module,
                           parallel_context: ParallelContext,
                           block_names: List[str]) -> PPWrapper:
    """
    Apply pipeline parallelism to a model.
    
    Args:
        model (nn.Module): The model to apply PP to
        parallel_context (ParallelContext): The parallel context
        block_names (List[str]): List of block container names to distribute across PP ranks
        
    Returns:
        PPWrapper: The wrapped model with PP applied
        
    Example:
        >>> # Initialize distributed and create context
        >>> dist.init_process_group(backend="nccl", init_method="env://")
        >>> ctx = ParallelContext({"pp": 4})
        >>> 
        >>> # Apply PP to model - only transformer blocks are distributed
        >>> model = GPT2Model(config)
        >>> pp_model = apply_pipeline_parallel(
        ...     model=model,
        ...     parallel_context=ctx,
        ...     block_names=["transformer.h"]
        ... )
        >>> 
        >>> # Use the PP model - forward pass automatically handles P2P communication
        >>> output = pp_model(input_ids)
    """
    return PPWrapper(
        model=model,
        parallel_context=parallel_context,
        block_names=block_names
    ) 