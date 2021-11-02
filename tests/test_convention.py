import time

import numpy as np
import pytest
import torch

from mmhuman3d.core.conventions.keypoints_mapping import (
    KEYPOINTS_FACTORY,
    convert_kps,
    get_flip_pairs,
    get_keypoint_idxs_by_part,
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
    original_mask[KEYPOINTS_FACTORY['smpl'].index('right_ankle')] = 0
    _, mask_coco = convert_kps(
        keypoints=keypoints, mask=original_mask, src='smpl', dst='coco')
    _, mask_coco_full = convert_kps(
        keypoints=keypoints, src='smpl', dst='coco')
    assert mask_coco[KEYPOINTS_FACTORY['coco'].index('right_ankle')] == 0
    mask_coco[KEYPOINTS_FACTORY['coco'].index('right_ankle')] = 1
    assert (mask_coco == mask_coco_full).all()

    # test approximate mapping
    keypoints = np.zeros((1, len(KEYPOINTS_FACTORY['smpl']), 3))
    _, mask_coco = convert_kps(
        keypoints=keypoints, src='smpl', dst='coco', approximate=False)
    _, approximate_mask_coco = convert_kps(
        keypoints=keypoints, src='smpl', dst='coco', approximate=True)
    assert mask_coco[KEYPOINTS_FACTORY['coco'].index('left_hip_extra')] == 0
    assert approximate_mask_coco[KEYPOINTS_FACTORY['coco'].index(
        'left_hip_extra')] == 1

    assert len(KEYPOINTS_FACTORY['human_data']) == len(
        set(KEYPOINTS_FACTORY['human_data']))


def test_cache():
    coco_wb_keypoints2d = np.ones((1, 133, 3))
    coco_wb_mask = np.ones((133, ))
    start_time = time.time()
    convert_kps(
        keypoints=coco_wb_keypoints2d,
        mask=coco_wb_mask,
        src='coco_wholebody',
        dst='smpl')
    without_cache_time = time.time() - start_time
    start_time = time.time()
    convert_kps(
        keypoints=coco_wb_keypoints2d,
        mask=coco_wb_mask,
        src='coco_wholebody',
        dst='smpl')
    with_cache_time = time.time() - start_time
    assert with_cache_time < without_cache_time


def test_get_flip_pairs():
    stable_conventions = [
        'coco', 'smpl', 'smplx', 'mpi_inf_3dhp', 'openpose_25'
    ]
    pair_len = [8, 9, 45, 10, 11]
    for idx in range(len(stable_conventions)):
        convention = stable_conventions[idx]
        flip_pairs = get_flip_pairs(convention)
        assert len(flip_pairs) == pair_len[idx]
        for flip_pair in flip_pairs:
            assert len(flip_pair) == 2
            assert type(flip_pair[0]) is int and type(flip_pair[1]) is int


def test_get_keypoint_idxs_by_part():
    stable_conventions = [
        'coco', 'smpl', 'smplx', 'mpi_inf_3dhp', 'openpose_25'
    ]
    head_len = [5, 1, 77, 2, 5]
    with pytest.raises(ValueError):
        head_idxs = get_keypoint_idxs_by_part('heed', stable_conventions[0])
    for idx in range(len(stable_conventions)):
        convention = stable_conventions[idx]
        head_idxs = get_keypoint_idxs_by_part('head', convention)
        assert len(head_idxs) == head_len[idx]
        for head_idx in head_idxs:
            assert type(head_idx) is int
