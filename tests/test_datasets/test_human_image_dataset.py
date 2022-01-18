import numpy as np

from mmhuman3d.data.datasets import HumanImageDataset


def test_human_image_dataset():
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
    test_dataset.num_data = 1
    outputs = [{
        'keypoints_3d': np.random.rand(num_data, 17, 3),
        'image_idx': np.arange(num_data)
    }]
    res = test_dataset.evaluate(outputs, res_folder='tests/data')
    assert 'MPJPE' in res
    assert 'MPJPE-PA' in res
    assert res['MPJPE'] > 0
    assert res['MPJPE-PA'] > 0

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
        'image_idx': np.arange(num_data)
    }]
    res = test_dataset.evaluate(outputs, res_folder='tests/data')
    assert 'MPJPE' in res
    assert 'MPJPE-PA' in res
    assert res['MPJPE'] > 0
    assert res['MPJPE-PA'] > 0

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
        'image_idx': np.arange(num_data)
    }]
    res = test_dataset.evaluate(outputs, res_folder='tests/data')
    assert 'MPJPE' in res
    assert 'MPJPE-PA' in res
    assert res['MPJPE'] > 0
    assert res['MPJPE-PA'] > 0


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
