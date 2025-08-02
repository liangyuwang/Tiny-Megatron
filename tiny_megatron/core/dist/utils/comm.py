# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.distributed as dist
import math

class ParallelContext:
    supported_parallel = ["tp", "dp", "pp"]
    _group_cache = {}  # avoid creating the same group multiple times

    def __init__(self, parallel_dims: dict[str, int]):
        assert dist.is_initialized(), "torch.distributed must be initialized first"

        self.parallel_dims = parallel_dims
        self.dim_names = list(parallel_dims.keys())
        self.dim_sizes = list(parallel_dims.values())
        self.ndim = len(self.dim_names)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert self.world_size == math.prod(self.dim_sizes), \
            f"World size {self.world_size} doesn't match product of dims {parallel_dims}"

        self.coords = self._compute_coords(self.rank, self.dim_sizes)
        self.axis_groups = self._build_groups()

    def _compute_coords(self, rank, dims):
        coords = []
        for dim in dims:
            coords.append(rank % dim)
            rank //= dim
        return coords

    def _coords_to_rank(self, coords):
        rank = 0
        multiplier = 1
        for dim, coord in zip(self.dim_sizes, coords):
            rank += coord * multiplier
            multiplier *= dim
        return rank

    def _build_groups(self):
        groups = {}
        
        # first, let all processes coordinate to create all required groups
        # this ensures that all processes call new_group() in the same order
        all_groups_to_create = set()
        
        # collect all groups to create
        for i, name in enumerate(self.dim_names):
            for coord_val in range(self.dim_sizes[i]):
                group_ranks = []
                for rank in range(self.world_size):
                    rank_coords = self._compute_coords(rank, self.dim_sizes)
                    if rank_coords[i] == coord_val:
                        group_ranks.append(rank)
                group_key = tuple(sorted(group_ranks))
                all_groups_to_create.add(group_key)
        
        # create all groups in a deterministic order (all processes do the same)
        for group_key in sorted(all_groups_to_create):
            if group_key not in ParallelContext._group_cache:
                group = dist.new_group(ranks=list(group_key))
                ParallelContext._group_cache[group_key] = group
        
        # now assign the correct group to the current process
        for i, name in enumerate(self.dim_names):
            current_coord_in_dim = self.coords[i]
            group_ranks = []
            
            for rank in range(self.world_size):
                rank_coords = self._compute_coords(rank, self.dim_sizes)
                if rank_coords[i] == current_coord_in_dim:
                    group_ranks.append(rank)
            
            group_key = tuple(sorted(group_ranks))
            groups[name] = ParallelContext._group_cache[group_key]
        
        return groups

    def get_group(self, name: str):
        return self.axis_groups[name]

    def get_rank_in(self, name: str):
        return self.coords[self.dim_names.index(name)]

    def get_coord_dict(self):
        return dict(zip(self.dim_names, self.coords))

    def print_debug(self):
        print(f"[Rank {self.rank}] coords: {self.get_coord_dict()}")
        for i, name in enumerate(self.dim_names):
            ranks_in_group = []
            current_coord_in_dim = self.coords[i]  # current rank's coordinate in this dimension
            
            for rank in range(self.world_size):
                rank_coords = self._compute_coords(rank, self.dim_sizes)
                # if the rank has the same coordinate in the current parallel dimension, it belongs to the same group
                if rank_coords[i] == current_coord_in_dim:
                    ranks_in_group.append(rank)
            print(f"  {name} group ranks: {ranks_in_group}")


if __name__=="__main__":
    # initialize distributed environment - force using gloo backend for testing
    dist.init_process_group(
        backend="gloo",  # use CPU backend to avoid GPU issues, can be changed to nccl if needed and you have enough GPUs
        init_method="env://"
    )
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"World size: {world_size}, Rank: {rank}")
    
    # select the appropriate configuration based on world_size
    if world_size == 8:
        config = {"tp": 2, "dp": 2, "pp": 2}
    elif world_size == 4:
        config = {"tp": 2, "dp": 2}
    elif world_size == 2:
        config = {"tp": 2, "dp": 1}
    else:
        print(f"Unsupported world_size: {world_size}")
        exit(1)
    
    ctx = ParallelContext(config)

    # except example: for 8 processes
    # Rank 0: {'tp': 0, 'dp': 0, 'pp': 0}
    # Rank 1: {'tp': 1, 'dp': 0, 'pp': 0} 
    # Rank 2: {'tp': 0, 'dp': 1, 'pp': 0}
    # Rank 3: {'tp': 1, 'dp': 1, 'pp': 0}
    # Rank 4: {'tp': 0, 'dp': 0, 'pp': 1}
    # Rank 5: {'tp': 1, 'dp': 0, 'pp': 1}
    # Rank 6: {'tp': 0, 'dp': 1, 'pp': 1}
    # Rank 7: {'tp': 1, 'dp': 1, 'pp': 1}
    # print(f"[Rank {rank}] Coordinates: {ctx.get_coord_dict()}")
    ctx.print_debug()
    
    dist.barrier()
    print(f"[Rank {rank}] Test completed!")