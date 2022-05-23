import numpy as np
import pytest
import torch

from mmhuman3d.models.heads.builder import PareHead


@pytest.mark.parametrize('deconv_with_bias', [True, False])
def test_pare_head(deconv_with_bias):

    # generate weight file for SMPL model.

    # initialize models
    head = PareHead(
        backbone='hrnet_w32-conv',
        use_keypoint_attention=True,
        smpl_mean_params='data/body_models/smpl_mean_params.npz',
        deconv_with_bias=deconv_with_bias)

    # mock inputs
    batch_size = 4
    input_shape = (batch_size, 480, 64, 64)
    features = _demo_head_inputs(input_shape)
    features = torch.tensor(features).float()

    predictions = head(features)
    pred_keys = ['pred_pose', 'pred_cam', 'pred_shape']

    for k in pred_keys:
        assert k in predictions
        assert predictions[k].shape[0] == batch_size


def test_pare_head_no_attention():

    # generate weight file for SMPL model.

    # initialize models
    head = PareHead(
        backbone='hrnet_w32-conv',
        use_keypoint_attention=False,
        use_heatmaps='',
        smpl_mean_params='data/body_models/smpl_mean_params.npz',
    )

    # mock inputs
    batch_size = 4
    input_shape = (batch_size, 480, 64, 64)
    features = _demo_head_inputs(input_shape)
    features = torch.tensor(features).float()

    predictions = head(features)
    pred_keys = ['pred_pose', 'pred_cam', 'pred_shape']

    for k in pred_keys:
        assert k in predictions
        assert predictions[k].shape[0] == batch_size


def _demo_head_inputs(input_shape=(1, 480, 56, 56)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    features = rng.rand(*input_shape)

    return features
