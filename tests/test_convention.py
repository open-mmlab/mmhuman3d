import time
import warnings

import numpy as np
import pytest
import torch

from mmhuman3d.core.conventions.keypoints_mapping import (
    KEYPOINTS_FACTORY,
    convert_kps,
    get_flip_pairs,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
    get_keypoint_num,
    get_mapping,
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
                # with mask
                keypoints_dst, mask = convert_kps(keypoints, src_name,
                                                  dst_name)

                # without mask
                keypoints_dst_wo_mask = convert_kps(
                    keypoints, src_name, dst_name, return_mask=False)

                assert np.all(keypoints_dst == keypoints_dst_wo_mask)

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


def test_duplicate_name():
    for data_name in KEYPOINTS_FACTORY:
        assert len(KEYPOINTS_FACTORY[data_name]) == len(
            set(KEYPOINTS_FACTORY[data_name]))


def test_approximate_mapping():

    # test SMPL_49 to SMPL_54
    SMPL_49_TO_SMPL_54 = [
        [
            0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 45, 46, 47, 48, 49, 50, 51, 52, 53
        ],
        [
            8, 12, 9, 29, 26, 30, 25, 1, 34, 33, 35, 32, 36, 31, 44, 46, 45,
            48, 47, 19, 20, 21, 22, 23, 24, 27, 28, 37, 38, 39, 40, 41, 42, 43
        ],
        [
            'pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'neck', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
            'right_wrist', 'nose', 'right_eye', 'left_eye', 'right_ear',
            'left_ear', 'left_bigtoe', 'left_smalltoe', 'left_heel',
            'right_bigtoe', 'right_smalltoe', 'right_heel', 'right_hip_extra',
            'left_hip_extra', 'neck_extra', 'headtop', 'pelvis_extra',
            'thorax_extra', 'spine_extra', 'jaw_extra', 'head_extra'
        ],
    ]
    mapped_result = get_mapping(src='smpl_49', dst='smpl_54', approximate=True)
    assert SMPL_49_TO_SMPL_54 == mapped_result

    # test SMPL_54 to SMPL_49
    SMPL_54_TO_SMPL_49 = [
        [i for i in range(49)],
        [
            24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18,
            20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27
        ],
        [
            'nose_openpose', 'neck_openpose', 'right_shoulder_openpose',
            'right_elbow_openpose', 'right_wrist_openpose',
            'left_shoulder_openpose', 'left_elbow_openpose',
            'left_wrist_openpose', 'pelvis_openpose', 'right_hip_openpose',
            'right_knee_openpose', 'right_ankle_openpose', 'left_hip_openpose',
            'left_knee_openpose', 'left_ankle_openpose', 'right_eye_openpose',
            'left_eye_openpose', 'right_ear_openpose', 'left_ear_openpose',
            'left_bigtoe_openpose', 'left_smalltoe_openpose',
            'left_heel_openpose', 'right_bigtoe_openpose',
            'right_smalltoe_openpose', 'right_heel_openpose', 'right_ankle',
            'right_knee', 'right_hip_extra', 'left_hip_extra', 'left_knee',
            'left_ankle', 'right_wrist', 'right_elbow', 'right_shoulder',
            'left_shoulder', 'left_elbow', 'left_wrist', 'neck_extra',
            'headtop', 'pelvis_extra', 'thorax_extra', 'spine_extra',
            'jaw_extra', 'head_extra', 'nose', 'left_eye', 'right_eye',
            'left_ear', 'right_ear'
        ],
    ]
    mapped_result = get_mapping(src='smpl_54', dst='smpl_49', approximate=True)
    assert SMPL_54_TO_SMPL_49 == mapped_result


def test_cache():
    coco_wb_keypoints2d = np.ones((1, 133, 3))
    coco_wb_mask = np.ones((133, ))
    start_time = time.time()
    # establish mapping cache at the first time
    for dst_key in KEYPOINTS_FACTORY:
        convert_kps(
            keypoints=coco_wb_keypoints2d,
            mask=coco_wb_mask,
            src='coco_wholebody',
            dst=dst_key)
    without_cache_time = time.time() - start_time
    start_time = time.time()
    # re-use cached mapping to convert faster
    for dst_key in KEYPOINTS_FACTORY:
        convert_kps(
            keypoints=coco_wb_keypoints2d,
            mask=coco_wb_mask,
            src='coco_wholebody',
            dst=dst_key)
    with_cache_time = time.time() - start_time
    if with_cache_time > without_cache_time:
        warnings.warn(
            'Cache doesn\'t reduce time spent on convention. '
            'Ignore this as a machine failure '
            'if convert_kps has not been modified.', UserWarning)


def test_get_flip_pairs():
    stable_conventions = [
        'coco', 'smpl', 'smplx', 'mpi_inf_3dhp', 'openpose_25'
    ]
    pair_len = [8, 9, 63, 10, 11]
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


def test_get_keypoint_idx():
    keypoint_names = \
        ['neck_openpose',
         'left_shoulder_openpose', 'right_shoulder_openpose',
         'left_hip_openpose', 'right_hip_openpose']

    smpl_49_idxs = [
        get_keypoint_idx(keypoint_name, 'smpl_49')
        for keypoint_name in keypoint_names
    ]

    assert smpl_49_idxs == [1, 5, 2, 12, 9]

    smpl_45_idxs = [
        get_keypoint_idx(keypoint_name, 'smpl_45', approximate=True)
        for keypoint_name in keypoint_names
    ]

    assert smpl_45_idxs == [12, 16, 17, 1, 2]


def test_get_keypoint_num():
    assert get_keypoint_num('smpl') == 24
    assert get_keypoint_num('smpl_24') == 24
    assert get_keypoint_num('smpl_45') == 45
    assert get_keypoint_num('smpl_49') == 49
    assert get_keypoint_num('smpl_54') == 54
