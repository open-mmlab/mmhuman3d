from typing import Optional, Union

import torch.nn as nn
from mmcv.runner.base_module import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class TemporalGRUEncoder(BaseModule):
    """TemporalEncoder used for VIBE. Adapted from
    https://github.com/mkocabas/VIBE.

    Args:
        input_size (int, optional): dimension of input feature. Default: 2048.
        num_layer (int, optional): number of layers for GRU. Default: 1.
        hidden_size (int, optional): hidden size for GRU. Default: 2048.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 input_size: Optional[int] = 2048,
                 num_layers: Optional[int] = 1,
                 hidden_size: Optional[int] = 2048,
                 init_cfg: Optional[Union[list, dict, None]] = None):
        super(TemporalGRUEncoder, self).__init__(init_cfg)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=num_layers)
        self.relu = nn.ReLU()
        self.linear = self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        N, T = x.shape[:2]
        x = x.permute(1, 0, 2)
        y, _ = self.gru(x)
        y = self.linear(self.relu(y).view(-1, self.hidden_size))
        y = y.view(T, N, self.input_size) + x
        y = y.permute(1, 0, 2).contiguous()
        return y
