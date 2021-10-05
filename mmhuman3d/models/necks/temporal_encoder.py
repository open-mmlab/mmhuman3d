import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS


@NECKS.register_module()
class TemporalGRUEncoder(nn.Module):
    """TemporalEncoder used for VIBE.

    Args:
        input_size (int, optional): Dimension of input feature. Default: 2048.
        num_layer (int, optional): Number of layers for GRU. Default: 1.
        hidden_size (int, optional): Hidden size for GRU. Default: 2048.
        bidirectional (bool, optional): Whether use bidirectional GRU or not.
            Default: False.
        add_linear (bool, optional): Whether use extra linear layer on the
            output of GRU module or not. Default: False.
        use_residual (bool, optional): Whether use residual connection in this
            module. Default: True.
    """

    def __init__(self,
                 input_size=2048,
                 num_layers=1,
                 hidden_size=2048,
                 bidirectional=False,
                 add_linear=False,
                 use_residual=True):
        super(TemporalGRUEncoder, self).__init__()

        self.input_size = input_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers)

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        N, T, L = x.shape
        x = x.permute(1, 0, 2)  # NTL -> TNL
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(T, N, L)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1, 0, 2).contiguous()  # TNL -> NTL
        return y
