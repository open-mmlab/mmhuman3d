import copy

import numpy as np

from mmhuman3d.data.datasets import HumanImageDataset
from mmhuman3d.data.datasets.pipelines import (
    LoadImageFromFile,
    MeshAffine,
    RandomHorizontalFlip,
)


def test_human_image_dataset():
    # test auto padding for bbox_xywh
    train_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='h36m',
        ann_file='sample_3dpw_test.npz')
    train_dataset.human_data.pop('bbox_xywh')
    data = train_dataset[0]
    assert sum(data['scale']) == 0
    assert sum(data['bbox_xywh']) == 0
    assert sum(data['center']) == 0

    train_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='h36m',
        ann_file='sample_3dpw_test.npz')
    data_keys = [
        'img_prefix', 'image_path', 'dataset_name', 'sample_idx', 'bbox_xywh',
        'center', 'scale', 'keypoints2d', 'keypoints3d', 'has_smpl',
        'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
    ]
    for i, data in enumerate(train_dataset):
        for key in data_keys:
            assert key in data

    num_data = 1
    test_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='pw3d',
        body_model=dict(
            type='SMPL',
            keypoint_src='smpl_45',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl'),
        ann_file='sample_3dpw_test.npz')
    test_dataset.num_data = num_data
    outputs = [{
        'keypoints_3d': np.random.rand(num_data, 17, 3),
        'smpl_pose': np.random.rand(num_data, 24, 3, 3),
        'smpl_beta': np.random.rand(num_data, 10),
        'image_idx': np.arange(num_data)
    }]
    res = test_dataset.evaluate(outputs, res_folder='tests/data')
    assert 'PA-MPJPE' in res
    assert res['PA-MPJPE'] > 0

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='mpjpe')
    assert 'MPJPE' in res
    assert res['MPJPE'] > 0

    res = test_dataset.evaluate(outputs, res_folder='tests/data', metric='pve')
    assert 'PVE' in res
    assert res['PVE'] > 0

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='pa-3dpck')
    assert 'PA-3DPCK' in res
    assert res['PA-3DPCK'] >= 0

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='3dpck')
    assert '3DPCK' in res
    assert res['3DPCK'] >= 0

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='pa-3dauc')
    assert 'PA-3DAUC' in res
    assert res['PA-3DAUC'] >= 0

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='3dauc')
    assert '3DAUC' in res
    assert res['3DAUC'] >= 0

    test_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='pw3d',
        body_model=dict(
            type='SMPL',
            keypoint_src='smpl_45',
            keypoint_dst='smpl_24',
            model_path='data/body_models/smpl'),
        ann_file='sample_3dpw_test.npz')
    test_dataset.num_data = 1
    outputs = [{
        'keypoints_3d': np.random.rand(num_data, 24, 3),
        'smpl_pose': np.random.rand(num_data, 24, 3, 3),
        'smpl_beta': np.random.rand(num_data, 10),
        'image_idx': np.arange(num_data)
    }]
    res = test_dataset.evaluate(outputs, res_folder='tests/data')
    assert 'PA-MPJPE' in res
    assert res['PA-MPJPE'] > 0

    test_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='pw3d',
        body_model=dict(
            type='SMPL',
            keypoint_src='smpl_45',
            keypoint_dst='smpl_49',
            model_path='data/body_models/smpl'),
        ann_file='sample_3dpw_test.npz')
    test_dataset.num_data = 1
    outputs = [{
        'keypoints_3d': np.random.rand(num_data, 49, 3),
        'smpl_pose': np.random.rand(num_data, 24, 3, 3),
        'smpl_beta': np.random.rand(num_data, 10),
        'image_idx': np.arange(num_data)
    }]
    res = test_dataset.evaluate(outputs, res_folder='tests/data')
    assert 'PA-MPJPE' in res
    assert res['PA-MPJPE'] > 0


def test_cached_dataset():
    # dump cache files
    train_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='h36m',
        cache_data_path='sample_3dpw_test_cache.npz',
        ann_file='sample_3dpw_test.npz')
    assert train_dataset.human_data is None

    # directly read from cache files
    train_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='h36m',
        cache_data_path='sample_3dpw_test_cache.npz',
        ann_file='sample_3dpw_test.npz')
    assert train_dataset.human_data is None

    # get data from cache files
    data_keys = [
        'img_prefix', 'image_path', 'dataset_name', 'sample_idx', 'bbox_xywh',
        'center', 'scale', 'keypoints2d', 'keypoints3d', 'has_smpl',
        'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
    ]
    for i, data in enumerate(train_dataset):
        for key in data_keys:
            assert key in data


def test_pipeline():
    train_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='3dpw',
        ann_file='sample_3dpw_train.npz')

    info = train_dataset.prepare_raw_data(0)

    # keypoints2d and 3d
    info['keypoints2d'] = np.random.rand(*info['keypoints2d'].shape).astype(
        np.float32)
    info['keypoints3d'] = np.random.rand(*info['keypoints3d'].shape).astype(
        np.float32)
    info['smpl_body_pose'] = info['smpl_body_pose'].astype('f')

    # test loading image
    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(info))

    # test no flip
    original_img = results['img']
    original_keypoints2d = results['keypoints2d']
    original_keypoints3d = results['keypoints3d']
    original_body_pose = results['smpl_body_pose']

    transform = RandomHorizontalFlip(flip_prob=0., convention='smpl_54')
    results_no_flip = transform(copy.deepcopy(results))
    assert np.equal(results_no_flip['img'], original_img).all()
    assert np.equal(results_no_flip['keypoints2d'], original_keypoints2d).all()
    assert np.equal(results_no_flip['keypoints3d'], original_keypoints3d).all()
    assert np.equal(results_no_flip['smpl_body_pose'],
                    original_body_pose).all()

    # test flip
    transform = RandomHorizontalFlip(flip_prob=1., convention='smpl_54')
    results_flip_smpl = transform(copy.deepcopy(results))
    assert not np.equal(results_flip_smpl['img'], original_img).all()
    assert not np.equal(results_flip_smpl['keypoints3d'],
                        original_keypoints3d).all()
    assert not np.equal(results_flip_smpl['keypoints2d'],
                        original_keypoints2d).all()
    assert not np.equal(results_flip_smpl['smpl_global_orient'],
                        original_body_pose).all()

    # test random affine
    transform = MeshAffine(img_res=224)
    results['rotation'] = 30
    results['scale'] = 0.25 * results['scale']
    results_affine = transform(copy.deepcopy(results))
    assert results_affine['img'].shape == (224, 224, 3)
    assert not np.equal(results_affine['img'].shape, original_img.shape).all()
    assert not np.equal(results_affine['keypoints3d'],
                        original_keypoints3d).all()
    assert not np.equal(results_affine['keypoints2d'],
                        original_keypoints2d).all()
    assert not np.equal(results_affine['smpl_global_orient'],
                        original_body_pose).all()


def test_human_image_dataset_smc():
    # test loading smc
    train_dataset = HumanImageDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='humman',
        ann_file='sample_humman_test_iphone_ds10.npz')

    data_keys = [
        'img_prefix', 'image_path', 'image_id', 'dataset_name', 'sample_idx',
        'bbox_xywh', 'center', 'scale', 'keypoints2d', 'keypoints3d',
        'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
        'smpl_transl'
    ]
    for i, data in enumerate(train_dataset):
        for key in data_keys:
            assert key in data
