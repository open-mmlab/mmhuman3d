import numpy as np
import torch

from mmhuman3d.models.heads.builder import HMRHead


def test_hmr_head():
    model = HMRHead(2048)
    # image feature from backbone
    x0_shape = (32, 2048, 7, 7)
    x0 = _demo_inputs(x0_shape)
    y0 = model(x0)
    assert y0['pred_pose'].shape == (32, 24, 3, 3)
    assert y0['pred_shape'].shape == (32, 10)
    assert y0['pred_cam'].shape == (32, 3)

    # image feature from multi-layer backbone
    x1_1_shape = (32, 1024, 14, 14)
    x1_2_shape = (32, 2048, 7, 7)
    x1 = [_demo_inputs(x1_1_shape), _demo_inputs(x1_2_shape)]
    y1 = model(x1)
    assert y1['pred_pose'].shape == (32, 24, 3, 3)
    assert y1['pred_shape'].shape == (32, 10)
    assert y1['pred_cam'].shape == (32, 3)

    # image feature from dataset
    x2_shape = (32, 2048)
    x2 = _demo_inputs(x2_shape)
    y2 = model(x2)
    assert y2['pred_pose'].shape == (32, 24, 3, 3)
    assert y2['pred_shape'].shape == (32, 10)
    assert y2['pred_cam'].shape == (32, 3)

    # video feature from dataset
    x3_shape = (32, 32, 2048)
    x3 = _demo_inputs(x3_shape)
    y3 = model(x3)
    assert y3['pred_pose'].shape == (32, 32, 24, 3, 3)
    assert y3['pred_shape'].shape == (32, 32, 10)
    assert y3['pred_cam'].shape == (32, 32, 3)


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run models.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    """
    feat = np.random.random(input_shape)
    feat = torch.FloatTensor(feat)

    return feat
