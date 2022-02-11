import copy
import os
import tempfile

import numpy as np
import pytest
import torch

from mmhuman3d.core.conventions.keypoints_mapping import get_flip_pairs
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets.builder import DATASETS
from mmhuman3d.data.datasets.pipelines import (
    Collect,
    GenerateHybrIKTarget,
    GetRandomScaleRotation,
    HybrIKAffine,
    HybrIKRandomFlip,
    ImageToTensor,
    LoadImageFromFile,
    NewKeypointsSelection,
    Normalize,
    RandomChannelNoise,
    RandomDPG,
    RandomOcclusion,
    ToTensor,
)
from mmhuman3d.data.datasets.pipelines.hybrik_transforms import (
    bbox_clip_xyxy,
    bbox_xywh_to_xyxy,
)
from mmhuman3d.utils.demo_utils import box2cs, xyxy2xywh


def get_3d_keypoints_vis(keypoints):
    keypoints, keypoints_vis = keypoints[:, :, :-1], keypoints[:, :, -1]
    num_datapoints, num_keypoints, dim = keypoints.shape
    joint_img = np.zeros((num_datapoints, num_keypoints, 3), dtype=np.float32)
    joint_vis = np.zeros((num_datapoints, num_keypoints, 3), dtype=np.float32)
    joint_img[:, :, :dim] = keypoints
    joint_vis[:, :, :dim] = np.tile(
        np.expand_dims(keypoints_vis, axis=2), (1, dim))
    return joint_img, joint_vis


def test_bbox():

    xywh = [521.16, 187.1, 28.59, 85.78]
    xywhc = [521.16, 187.1, 28.59, 85.78, 1.0]

    with pytest.raises(IndexError):
        _ = bbox_xywh_to_xyxy(xywhc)

    with pytest.raises(IndexError):
        _ = bbox_xywh_to_xyxy(np.array([xywhc] * 2))

    with pytest.raises(TypeError):
        _ = bbox_xywh_to_xyxy(torch.tensor(xywh))

    xyxy = bbox_xywh_to_xyxy(xywh)
    assert isinstance(xyxy, tuple)
    assert len(xyxy) == 4

    xyxy = bbox_xywh_to_xyxy(np.array([xywh] * 2))
    assert isinstance(xyxy, np.ndarray)
    assert len(xyxy) == 2
    assert len(xyxy[0]) == 4

    xyxy = [521.16, 187.1, 548.75, 271.88]
    xyxyc = [521.16, 187.1, 548.75, 271.88, 1.0]
    width, height = 640, 427

    with pytest.raises(IndexError):
        _ = bbox_clip_xyxy(xyxyc, width, height)

    with pytest.raises(IndexError):
        _ = bbox_clip_xyxy(np.array([xyxyc]), width, height)

    with pytest.raises(TypeError):
        _ = bbox_clip_xyxy(torch.tensor(xywh), width, height)

    xyxy = bbox_clip_xyxy(xyxy, width, height)
    assert isinstance(xyxy, tuple)
    assert len(xyxy) == 4

    xyxy = bbox_clip_xyxy(np.array([xyxy]), width, height)
    assert isinstance(xyxy, np.ndarray)
    assert len(xyxy) == 4


def _load_test_data():
    ann_file = 'tests/data/preprocessed_datasets/h36m_hybrik_train.npz'
    img_prefix = 'tests/data/datasets/h36m'
    index = 0

    data = HumanData()
    data.load(ann_file)

    info = {}
    info['ann_info'] = {}
    info['img_prefix'] = None
    info['image_path'] = os.path.join(img_prefix, data['image_path'][index])

    bbox_xyxy = data['bbox_xywh'][index]
    info['bbox'] = bbox_xyxy[:4]
    bbox_xywh = xyxy2xywh(bbox_xyxy)
    center, scale = box2cs(bbox_xywh, aspect_ratio=1.0, bbox_scale_factor=1.25)

    info['center'] = center
    info['scale'] = scale
    info['rotation'] = 0

    info['ann_info']['dataset_name'] = 'h36m'
    info['ann_info']['height'] = data['image_height'][index]
    info['ann_info']['width'] = data['image_width'][index]
    info['depth_factor'] = float(data['depth_factor'][index])

    try:
        keypoints3d, keypoints3d_vis = get_3d_keypoints_vis(
            data['keypoints2d'])
    except KeyError:
        keypoints3d, keypoints3d_vis = get_3d_keypoints_vis(
            data['keypoints3d'])

    info['keypoints3d'] = keypoints3d[index]
    info['keypoints3d_vis'] = keypoints3d_vis[index]

    try:
        smpl = data['smpl']
        if 'has_smpl' not in data.keys():
            info['has_smpl'] = 1
        else:
            info['has_smpl'] = data['has_smpl'].astype(np.float32)
        keypoints3d_relative, _ = get_3d_keypoints_vis(
            data['keypoints3d_relative'])
        keypoints3d17, keypoints3d17_vis = \
            get_3d_keypoints_vis(data['keypoints3d17'])
        keypoints3d17_relative, _ = get_3d_keypoints_vis(
            data['keypoints3d17_relative'])
    except KeyError:
        info['has_smpl'] = 0

    if info['has_smpl']:
        info['pose'] = smpl['thetas'][index].astype(np.float32)
        info['beta'] = smpl['betas'][index].astype(np.float32)
        info['keypoints3d_relative'] = keypoints3d_relative[index]
        info['keypoints3d17'] = keypoints3d17[index]
        info['keypoints3d17_vis'] = keypoints3d17_vis[index]
        info['keypoints3d17_relative'] = keypoints3d17_relative[index]

    try:
        info['intrinsic_param'] = data['cam_param']['intrinsic'][index].astype(
            np.float32)
    except KeyError:
        info['intrinsic_param'] = np.zeros((3, 3))

    try:
        info['joint_root'] = data['root_cam'][index].astype(np.float32)
    except KeyError:
        info['joint_root'] = np.zeros((1, 3))

    try:
        info['target_twist'] = data['phi'][index].astype(np.float32)
        info['target_twist_weight'] = data['phi_weight'][index].astype(
            np.float32)
    except KeyError:
        info['target_twist'] = np.zeros((23, 2))
        info['target_twist_weight'] = np.zeros_like((info['target_twist']))

    info['sample_idx'] = 0
    return copy.deepcopy(info)


def test_hybrik_pipeline():
    results = _load_test_data()

    # test loading image
    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(results))
    assert results['img'].shape == (1002, 1000, 3)

    # test random dpg
    transform = RandomDPG(dpg_prob=1.)
    results_dpg = transform(copy.deepcopy(results))

    assert not np.equal(results_dpg['bbox'], results['bbox']).all()
    assert not np.equal(results_dpg['center'], results['center']).all()
    assert not np.equal(results_dpg['scale'], results['scale']).all()

    # test random scale rotate
    transform = GetRandomScaleRotation(
        rot_factor=30, scale_factor=0.25, rot_prob=1.)
    results_scalerot = transform(copy.deepcopy(results))

    assert not np.equal(results_scalerot['rotation'],
                        results['rotation']).all()
    assert not np.equal(results_scalerot['scale'], results['scale']).all()

    # test random occlusion
    transform = RandomOcclusion(occlusion_prob=1.)
    results_occlusion = transform(copy.deepcopy(results))

    assert not np.equal(results_occlusion['img'], results['img']).all()

    # test random flip
    humandata_flip_pairs = get_flip_pairs('human_data')
    original_img = results['img']
    original_twist = results['target_twist']
    original_keypoints3d17_relative = results['keypoints3d17_relative']
    original_keypoints3d_relative = results['keypoints3d_relative']
    original_keypoints3d17 = results['keypoints3d17']
    original_pose = results['pose']
    original_keypoints3d = results['keypoints3d']

    # test no flip
    transform = HybrIKRandomFlip(flip_prob=0., flip_pairs=humandata_flip_pairs)
    results_no_flip = transform(copy.deepcopy(results))
    assert np.equal(results_no_flip['img'], original_img).all()
    assert np.equal(results_no_flip['keypoints3d'], original_keypoints3d).all()

    # test flip
    transform = HybrIKRandomFlip(
        flip_prob=1.0, flip_pairs=humandata_flip_pairs)

    # test flip for data without smpl
    results['has_smpl'] = 0
    results_flip = transform(copy.deepcopy(results))
    assert not np.equal(results_flip['img'], original_img).all()
    assert np.equal(results_flip['target_twist'], original_twist).all()
    assert np.equal(results_flip['keypoints3d17_relative'],
                    original_keypoints3d17_relative).all()
    assert np.equal(results_flip['keypoints3d17'],
                    original_keypoints3d17).all()
    assert np.equal(results_flip['keypoints3d_relative'],
                    original_keypoints3d_relative).all()
    assert np.equal(results_flip['pose'], original_pose).all()
    assert not np.equal(results_flip['keypoints3d'],
                        original_keypoints3d).all()

    # test flip for data containing smpl
    results['has_smpl'] = 1
    results_flip_smpl = transform(copy.deepcopy(results))
    assert not np.equal(results_flip_smpl['img'], original_img).all()
    assert not np.equal(results_flip_smpl['target_twist'],
                        original_twist).all()
    assert not np.equal(results_flip_smpl['keypoints3d17_relative'],
                        original_keypoints3d_relative).all()
    assert not np.equal(results_flip_smpl['keypoints3d17'],
                        original_keypoints3d17).all()
    assert not np.equal(results_flip_smpl['pose'], original_pose).all()
    assert not np.equal(results_flip_smpl['keypoints3d'],
                        original_keypoints3d).all()
    assert not np.equal(results_flip_smpl['keypoints3d_relative'],
                        original_keypoints3d_relative).all()

    # test keypoints selection
    h36m_idxs = [
        148, 145, 4, 7, 144, 5, 8, 150, 146, 152, 147, 16, 18, 20, 17, 19, 21
    ]
    hybrik29_idxs = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 16, 17, 18, 19,
        20, 21, 66, 71, 15, 68, 73, 60, 63
    ]
    keypoints_maps = [
        dict(
            keypoints=[
                'keypoints3d17',
                'keypoints3d17_vis',
                'keypoints3d17_relative',
            ],
            keypoints_index=h36m_idxs),
        dict(
            keypoints=[
                'keypoints3d', 'keypoints3d_vis', 'keypoints3d_relative'
            ],
            keypoints_index=hybrik29_idxs),
    ]
    transform = NewKeypointsSelection(maps=keypoints_maps)
    results = transform(copy.deepcopy(results))
    assert len(results['keypoints3d17']) == len(h36m_idxs)
    assert len(results['keypoints3d17_vis']) == len(h36m_idxs)
    assert len(results['keypoints3d17_relative']) == len(h36m_idxs)
    assert len(results['keypoints3d']) == len(hybrik29_idxs)
    assert len(results['keypoints3d_vis']) == len(hybrik29_idxs)
    assert len(results['keypoints3d_relative']) == len(hybrik29_idxs)
    assert np.equal(results['keypoints3d17_vis'], np.ones([len(h36m_idxs),
                                                           3])).all()
    assert np.equal(results['keypoints3d_vis'],
                    np.ones([len(hybrik29_idxs), 3])).all()

    # test random affine
    transform = HybrIKAffine(img_res=224)

    # test affine for data without smpl
    results['has_smpl'] = 0
    original_keypoints3d17 = results['keypoints3d17']
    original_keypoints3d = results['keypoints3d']

    results_affine = transform(copy.deepcopy(results))
    assert results_affine['img'].shape == (224, 224, 3)
    assert np.equal(results_affine['keypoints3d17'],
                    original_keypoints3d17).all()
    assert not np.equal(results_affine['keypoints3d'],
                        original_keypoints3d).all()

    # test affine for data with smpl
    results['has_smpl'] = 1

    results = transform(copy.deepcopy(results))
    assert results['img'].shape == (224, 224, 3)
    assert not np.equal(results['keypoints3d17'], original_keypoints3d17).all()
    assert not np.equal(results['keypoints3d'], original_keypoints3d).all()

    new_keys = [
        'target_uvd_29', 'target_xyz_24', 'target_weight_24',
        'target_weight_29', 'target_xyz_17', 'target_weight_17',
        'target_theta', 'target_beta', 'target_smpl_weight',
        'target_theta_weight'
    ]
    for k in new_keys:
        assert k not in results

    # test generate hybrik target
    transform = GenerateHybrIKTarget(img_res=256, test_mode=False)

    # mock coco
    results_coco = copy.deepcopy(results)
    results_coco['has_smpl'] = 0
    results_coco['ann_info']['dataset_name'] = 'coco'
    results_coco = transform(copy.deepcopy(results_coco))

    # mock mpi_inf_3dhp
    results_hp3d = copy.deepcopy(results)
    results_hp3d['has_smpl'] = 0
    results_hp3d['ann_info']['dataset_name'] = 'mpi_inf_3dhp'
    results_hp3d = transform(copy.deepcopy(results_hp3d))

    # moch h36m
    results = transform(copy.deepcopy(results))

    for k in new_keys:
        assert k in results
        assert k in results_coco
        assert k in results_hp3d

    # test channel noise
    random_noise = RandomChannelNoise(noise_factor=0.2)
    results_noise = random_noise(copy.deepcopy(results))
    assert not np.equal(results['img'], results_noise['img']).all()

    # test norm
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)

    normalize = Normalize(**img_norm_cfg)

    results_normalize = normalize(copy.deepcopy(results))
    assert not np.equal(results_normalize['img'], results['img']).all()

    # transfer image to tensor
    img_to_tensor = ImageToTensor(keys=['img'])
    results_tensor = img_to_tensor(copy.deepcopy(results))
    assert isinstance(results_tensor['img'], torch.Tensor)
    assert results_tensor['img'].shape == torch.Size([3, 224, 224])

    # transfer keys to tensor
    data_keys = [
        'trans_inv', 'intrinsic_param', 'joint_root', 'depth_factor',
        'target_uvd_29', 'target_xyz_24', 'target_weight_24',
        'target_weight_29', 'target_xyz_17', 'target_weight_17',
        'target_theta', 'target_beta', 'target_smpl_weight',
        'target_theta_weight', 'target_twist', 'target_twist_weight', 'bbox',
        'sample_idx'
    ]
    for k in data_keys:
        assert not isinstance(results[k], torch.Tensor)
    to_tensor = ToTensor(keys=data_keys)
    results_tensor = to_tensor(copy.deepcopy(results))
    for k in data_keys:
        assert isinstance(results_tensor[k], torch.Tensor)

    # test collect
    meta_keys = ['center', 'scale', 'rotation', 'image_path']
    collect = Collect(keys=data_keys, meta_keys=meta_keys)
    results_final = collect(results_normalize)

    for k in data_keys:
        assert k in results_final
    for k in meta_keys:
        assert k in results_final['img_metas'].data


def test_human_hybrik_dataset():
    # test HumanHybrikDataset

    dataset = 'HybrIKHumanImageDataset'
    dataset_class = DATASETS.get(dataset)

    body_model = dict(type='SMPL', model_path='data/body_models/smpl')
    # train mode
    custom_dataset = dataset_class(
        dataset_name='h36m',
        data_prefix='tests/data',
        body_model=body_model,
        pipeline=[],
        ann_file='h36m_hybrik_train.npz')

    keys = [
        'ann_info', 'image_path', 'bbox', 'center', 'scale', 'rotation',
        'depth_factor', 'has_smpl', 'joint_root', 'intrinsic_param',
        'target_twist', 'target_twist_weight', 'keypoints3d',
        'keypoints3d_vis', 'pose', 'beta', 'keypoints3d_relative',
        'keypoints3d17', 'keypoints3d17_vis', 'keypoints3d17_relative',
        'dataset_name', 'sample_idx'
    ]
    sample_item = custom_dataset[0]
    for k in keys:
        assert k in sample_item
    body_model = dict(type='SMPL', model_path='data/body_models/smpl')
    # test mode
    num_data = 1
    custom_dataset = dataset_class(
        dataset_name='h36m',
        data_prefix='tests/data',
        body_model=body_model,
        pipeline=[],
        ann_file='h36m_hybrik_train.npz',
        test_mode=True)
    custom_dataset.num_data = num_data
    # test evaluation
    outputs = [{
        'xyz_17': np.random.rand(num_data, 17, 3),
        'smpl_pose': np.random.rand(num_data, 24, 3, 3),
        'smpl_beta': np.random.rand(num_data, 10),
        'image_idx': np.arange(num_data)
    }]
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_result = custom_dataset.evaluate(outputs, tmpdir)
        assert 'PA-MPJPE' in eval_result
        assert eval_result['PA-MPJPE'] > 0

        res = custom_dataset.evaluate(
            outputs, res_folder=tmpdir, metric='mpjpe')
        assert 'MPJPE' in res
        assert res['MPJPE'] > 0

        res = custom_dataset.evaluate(
            outputs, res_folder=tmpdir, metric='pa-3dpck')
        assert 'PA-3DPCK' in res
        assert res['PA-3DPCK'] >= 0

        res = custom_dataset.evaluate(
            outputs, res_folder=tmpdir, metric='3dpck')
        assert '3DPCK' in res
        assert res['3DPCK'] >= 0

        res = custom_dataset.evaluate(
            outputs, res_folder=tmpdir, metric='pa-3dauc')
        assert 'PA-3DAUC' in res
        assert res['PA-3DAUC'] >= 0

        res = custom_dataset.evaluate(
            outputs, res_folder=tmpdir, metric='3dauc')
        assert '3DAUC' in res
        assert res['3DAUC'] >= 0
