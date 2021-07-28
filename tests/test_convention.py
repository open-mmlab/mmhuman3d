import numpy as np
import pytest
import torch

from mmhuman3d.core.conventions.joints_mapping.kp_mapping import (
    JOINTS_FACTORY,
    convert_kps,
)


def test_conventions():
    data_names = list(JOINTS_FACTORY.keys())
    f = 10
    n_person = 3
    # name should be in JOINTS_FACTORY
    with pytest.raises(KeyError):
        joints_dst, mask = convert_kps(np.zeros((f, 17, 3)), '1', '2')
    # shape of joints should be (f * J * 3/2) or (f * n * K * 3/2)
    with pytest.raises(AssertionError):
        joints_dst, mask = convert_kps(np.zeros((17, 3)), 'coco', 'coco')
    for src_name in data_names:
        for dst_name in data_names:
            J = len(JOINTS_FACTORY[src_name])
            J_dst = len(JOINTS_FACTORY[dst_name])

            # test joints3d/joints2d input as numpy
            for joints in [
                    np.zeros((f, J, 3)),
                    np.zeros((f, J, 2)),
                    np.zeros((f, n_person, J, 3)),
                    np.zeros((f, n_person, J, 2))
            ]:
                joints_dst, mask = convert_kps(joints, src_name, dst_name)
                exp_shape = list(joints.shape)
                exp_shape[-2] = J_dst
                assert joints_dst.shape == tuple(exp_shape)
                assert mask.shape == (J_dst, )
                if src_name == dst_name:
                    assert mask.all() == 1
            # test joints3d/joints2d input as tensor
            for joints in [
                    torch.zeros((f, J, 3)),
                    torch.zeros((f, J, 2)),
                    torch.zeros((f, n_person, J, 3)),
                    torch.zeros((f, n_person, J, 2))
            ]:
                joints_dst, mask = convert_kps(joints, src_name, dst_name)
                exp_shape = list(joints.shape)
                exp_shape[-2] = J_dst
                assert joints_dst.shape == torch.Size(exp_shape)
                assert mask.shape == torch.Size([J_dst])
                if src_name == dst_name:
                    assert mask.all() == 1
