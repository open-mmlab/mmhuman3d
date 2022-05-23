import torch

from mmhuman3d.models.necks.temporal_encoder import TemporalGRUEncoder


def test_hmr_head():
    model = TemporalGRUEncoder(2048)
    x = torch.rand(32, 32, 2048)
    y = model(x)
    assert y.shape == (32, 32, 2048)
