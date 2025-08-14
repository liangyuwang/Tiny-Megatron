# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch.nn as nn

class Parameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, dp_bwd_sync=False):
        t = nn.Parameter.__new__(cls, data, requires_grad)
        t.dp_bwd_sync = dp_bwd_sync
        return t

