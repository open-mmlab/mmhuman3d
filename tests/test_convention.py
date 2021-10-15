import time

import numpy as np
import pytest
import torch

from mmhuman3d.core.conventions.keypoints_mapping import (
    KEYPOINTS_FACTORY,
    convert_kps,
)


def test_conventions():
    data_names = list(KEYPOINTS_FACTORY.keys())
    f = 10
    n_person = 3
    # name should be in KEYPOINTS_FACTORY
    with pytest.raises(KeyError):
        keypoints_dst, mask = convert_kps(np.zeros((f, 17, 3)), '1', '2')
    # shape of keypoints should be (f * J * 3/2) or (f * n * K * 3/2)
    with pytest.raises(AssertionError):
        keypoints_dst, mask = convert_kps(np.zeros((17, 3)), 'coco', 'coco')
    for src_name in data_names:
        for dst_name in data_names:
            J = len(KEYPOINTS_FACTORY[src_name])
            J_dst = len(KEYPOINTS_FACTORY[dst_name])

            # test keypoints3d/keypoints2d input as numpy
            for keypoints in [
                    np.zeros((f, J, 3)),
                    np.zeros((f, J, 2)),
                    np.zeros((f, n_person, J, 3)),
                    np.zeros((f, n_person, J, 2))
            ]:
                keypoints_dst, mask = convert_kps(keypoints, src_name,
                                                  dst_name)
                exp_shape = list(keypoints.shape)
                exp_shape[-2] = J_dst
                assert keypoints_dst.shape == tuple(exp_shape)
                assert mask.shape == (J_dst, )
                if src_name == dst_name:
                    assert mask.all() == 1
            # test keypoints3d/keypoints2d input as tensor
            for keypoints in [
                    torch.zeros((f, J, 3)),
                    torch.zeros((f, J, 2)),
                    torch.zeros((f, n_person, J, 3)),
                    torch.zeros((f, n_person, J, 2))
            ]:
                keypoints_dst, mask = convert_kps(keypoints, src_name,
                                                  dst_name)
                exp_shape = list(keypoints.shape)
                exp_shape[-2] = J_dst
                assert keypoints_dst.shape == torch.Size(exp_shape)
                assert mask.shape == torch.Size([J_dst])
                if src_name == dst_name:
                    assert mask.all() == 1

    # test original_mask
    keypoints = np.zeros((1, len(KEYPOINTS_FACTORY['smpl']), 3))
    original_mask = np.ones((len(KEYPOINTS_FACTORY['smpl'])))
    original_mask[KEYPOINTS_FACTORY['smpl'].index('left_hip')] = 0
    _, mask_coco = convert_kps(
        keypoints=keypoints, mask=original_mask, src='smpl', dst='coco')
    _, mask_coco_full = convert_kps(
        keypoints=keypoints, src='smpl', dst='coco')
    assert mask_coco[KEYPOINTS_FACTORY['coco'].index('left_hip')] == 0
    mask_coco[KEYPOINTS_FACTORY['coco'].index('left_hip')] = 1
    assert (mask_coco == mask_coco_full).all()


def test_cache():
    mmpose_keypoints2d = np.ones((1, 133, 3))
    mmpose_mask = np.ones((133, ))
    start_time = time.time()
    convert_kps(
        keypoints=mmpose_keypoints2d,
        mask=mmpose_mask,
        src='mmpose',
        dst='smpl')
    without_cache_time = time.time() - start_time
    start_time = time.time()
    convert_kps(
        keypoints=mmpose_keypoints2d,
        mask=mmpose_mask,
        src='mmpose',
        dst='smpl')
    with_cache_time = time.time() - start_time
    assert with_cache_time < without_cache_time
