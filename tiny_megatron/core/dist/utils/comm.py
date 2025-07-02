# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.distributed as dist
import math

class ParallelContext:
    supported_parallel = ["tp", "dp", "pp"]

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
        for dim in reversed(dims):
            coords.append(rank % dim)
            rank //= dim
        return list(reversed(coords))

    def _coords_to_rank(self, coords):
        rank = 0
        multiplier = 1
        for dim, coord in zip(reversed(self.dim_sizes), reversed(coords)):
            rank += coord * multiplier
            multiplier *= dim
        return rank

    def _build_groups(self):
        groups = {}
        for i, name in enumerate(self.dim_names):
            group_ranks = []
            for j in range(self.dim_sizes[i]):
                coords = list(self.coords)
                coords[i] = j
                rank = self._coords_to_rank(coords)
                group_ranks.append(rank)
            group = dist.new_group(ranks=group_ranks)
            groups[name] = group
        return groups

    def get_group(self, name: str):
        return self.axis_groups[name]

    def get_rank_in(self, name: str):
        return self.coords[self.dim_names.index(name)]

    def get_coord_dict(self):
        return dict(zip(self.dim_names, self.coords))

    def print_debug(self):
        print(f"[Rank {self.rank}] coords: {self.get_coord_dict()}")
        for name in self.dim_names:
            print(f"  {name} group: {self.axis_groups[name]}")


if __name__=="__main__":
    dist.init_process_group("nccl")
    ctx = ParallelContext({"tp": 2, "pp": 2, "dp": 2})

    tp_group = ctx.get_group("tp")
    tp_rank = ctx.get_rank_in("tp")

    print(ctx.get_coord_dict())  # {'tp': 0, 'pp': 1, 'dp': 1}