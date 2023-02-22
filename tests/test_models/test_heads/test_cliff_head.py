import numpy as np
import torch

from mmhuman3d.models.heads.builder import CliffHead


def test_cliff_head():
    # initialize models
    model = CliffHead(
        feat_dim=2048,
        smpl_mean_params='data/body_models/smpl_mean_params.npz')

    bbox_info = torch.FloatTensor([[-0.5, 0.2, 1.5]])
    # image feature from backbone
    batch_size = 32
    x0_shape = (batch_size, 2048, 7, 7)
    x0 = _demo_head_inputs(x0_shape)
    x0 = torch.tensor(x0).float()
    y0 = model(x0, bbox_info)
    assert y0['pred_pose'].shape == (batch_size, 24, 3, 3)
    assert y0['pred_shape'].shape == (batch_size, 10)
    assert y0['pred_cam'].shape == (batch_size, 3)

    # image feature from multi-layer backbone
    x1_1_shape = (batch_size, 1024, 14, 14)
    x1_2_shape = (batch_size, 2048, 7, 7)
    x1 = [_demo_head_inputs(x1_1_shape), _demo_head_inputs(x1_2_shape)]
    y1 = model(x1, bbox_info)
    assert y1['pred_pose'].shape == (batch_size, 24, 3, 3)
    assert y1['pred_shape'].shape == (batch_size, 10)
    assert y1['pred_cam'].shape == (batch_size, 3)


def _demo_head_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run models.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    """
    features = np.random.random(input_shape)
    features = torch.FloatTensor(features)

    return features
