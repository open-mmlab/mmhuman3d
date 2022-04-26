import os
import os.path as osp
import tempfile

import numpy as np
import pytest
import torch

from mmhuman3d.models.heads import PareHead


def generate_weights(output_dir):
    """Generate a SMPL model weight file to initialize SMPL model, and generate
    a 3D joints regressor file."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joint_regressor_file = os.path.join(output_dir, 'J_regressor_h36m.npy')
    np.save(joint_regressor_file, np.zeros([17, 6890]))

    joint_extra_file = os.path.join(output_dir, 'J_regressor_extra.npy')
    np.save(joint_extra_file, np.zeros([9, 6890]))
    smpl_mean = {
        'shape': np.zeros(10),
        'pose': np.zeros(144),
        'cam': np.zeros(3)
    }
    smpl_mean_file = os.path.join(output_dir, 'h36m_mean.npz')
    np.savez(smpl_mean_file, **smpl_mean)
    return


@pytest.mark.parametrize('deconv_with_bias', [True, False])
def test_PARE_head(deconv_with_bias):

    tmpdir = tempfile.TemporaryDirectory()
    # generate weight file for SMPL model.
    generate_weights(tmpdir.name)

    # initialize models
    head = PareHead(
        backbone='hrnet_w32-conv',
        use_keypoint_attention=True,
        smpl_mean_params=osp.join(tmpdir.name, 'h36m_mean.npz'),
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

    tmpdir.cleanup()


def test_PARE_head_no_attention():

    tmpdir = tempfile.TemporaryDirectory()
    # generate weight file for SMPL model.
    generate_weights(tmpdir.name)

    # initialize models
    head = PareHead(
        backbone='hrnet_w32-conv',
        use_keypoint_attention=False,
        use_heatmaps='',
        smpl_mean_params=osp.join(tmpdir.name, 'h36m_mean.npz'),
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

    tmpdir.cleanup()


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
