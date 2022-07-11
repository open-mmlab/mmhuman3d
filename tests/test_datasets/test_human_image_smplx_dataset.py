import copy

import numpy as np

from mmhuman3d.data.datasets import HumanImageSMPLXDataset
from mmhuman3d.data.datasets.pipelines import (
    BBoxCenterJitter,
    LoadImageFromFile,
    MeshAffine,
    RandomChannelNoise,
    RandomHorizontalFlip,
    Rotation,
    SimulateLowRes,
)

face_vertex_ids_path = 'data/body_models/smplx/SMPL-X__FLAME_vertex_ids.npy'
hand_vertex_ids_path = 'data/body_models/smplx/MANO_SMPLX_vertex_ids.pkl'


def test_human_image_smplx_dataset():
    train_dataset = HumanImageSMPLXDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='EHF',
        ann_file='sample_ehf_val.npz')
    data_keys = [
        'img_prefix', 'image_path', 'dataset_name', 'sample_idx', 'bbox_xywh',
        'center', 'scale', 'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
        'has_smplx_global_orient', 'has_smplx_body_pose', 'has_smplx_jaw_pose',
        'has_smplx_right_hand_pose', 'has_smplx_left_hand_pose',
        'has_smplx_betas', 'has_smplx_expression', 'smplx_jaw_pose',
        'smplx_body_pose', 'smplx_right_hand_pose', 'smplx_left_hand_pose',
        'smplx_global_orient', 'smplx_betas', 'keypoints2d', 'keypoints3d',
        'smplx_expression'
    ]
    for i, data in enumerate(train_dataset):
        for key in data_keys:
            assert key in data

    test_dataset = HumanImageSMPLXDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='EHF',
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            model_path='data/body_models/smplx'),
        ann_file='sample_ehf_val.npz',
        face_vertex_ids_path=face_vertex_ids_path,
        hand_vertex_ids_path=hand_vertex_ids_path,
        convention='smplx')
    num_data = 1
    test_dataset.num_data = num_data
    outputs = [{
        'keypoints_3d': np.random.rand(num_data, 144, 3),
        'vertices': np.random.rand(num_data, 10475, 3),
        'image_idx': np.arange(num_data)
    }]

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='pa-mpjpe')
    assert 'PA-MPJPE' in res
    assert res['PA-MPJPE'] > 0

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='mpjpe')
    assert 'MPJPE' in res
    assert res['MPJPE'] > 0

    res = test_dataset.evaluate(
        outputs, res_folder='tests/data', metric='pa-pve')
    assert 'PA-PVE' in res
    assert res['PA-PVE'] > 0

    res = test_dataset.evaluate(
        outputs,
        res_folder='tests/data',
        metric=['pa-mpjpe'],
        body_part=[['body', 'right_hand', 'left_hand']])
    assert 'BODY PA-MPJPE' in res
    assert 'RIGHT_HAND PA-MPJPE' in res
    assert 'LEFT_HAND PA-MPJPE' in res
    assert res['BODY PA-MPJPE'] > 0
    assert res['RIGHT_HAND PA-MPJPE'] > 0
    assert res['LEFT_HAND PA-MPJPE'] > 0

    res = test_dataset.evaluate(
        outputs,
        res_folder='tests/data',
        metric=['pa-pve'],
        body_part=[['', 'right_hand', 'left_hand', 'face']])
    assert 'PA-PVE' in res
    assert 'FACE PA-PVE' in res
    assert 'RIGHT_HAND PA-PVE' in res
    assert 'LEFT_HAND PA-PVE' in res
    assert res['FACE PA-PVE'] > 0
    assert res['RIGHT_HAND PA-PVE'] > 0
    assert res['LEFT_HAND PA-PVE'] > 0


def test_pipeline():
    train_dataset = HumanImageSMPLXDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='3dpw',
        ann_file='sample_3dpw_train.npz',
        convention='smplx')
    info = train_dataset.prepare_raw_data(0)

    info['keypoints2d'] = np.random.rand(*info['keypoints2d'].shape).astype(
        np.float32)
    info['keypoints3d'] = np.random.rand(*info['keypoints3d'].shape).astype(
        np.float32)
    info['smplx_body_pose'] = np.random.rand(
        *info['smplx_body_pose'].shape).astype(np.float32)

    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(info))

    # test no flip
    original_img = results['img']
    original_keypoints2d = results['keypoints2d']
    original_keypoints3d = results['keypoints3d']
    original_body_pose = results['smplx_body_pose']
    transform = RandomHorizontalFlip(flip_prob=0., convention='smplx')
    results_no_flip = transform(copy.deepcopy(results))
    assert np.equal(results_no_flip['img'], original_img).all()
    assert np.equal(results_no_flip['keypoints2d'], original_keypoints2d).all()
    assert np.equal(results_no_flip['keypoints3d'], original_keypoints3d).all()
    assert np.equal(results_no_flip['smplx_body_pose'],
                    original_body_pose).all()

    # test flip
    transform = RandomHorizontalFlip(flip_prob=1., convention='smplx')
    results_flip = transform(copy.deepcopy(results))
    assert not np.equal(results_flip['img'], original_img).all()
    assert not np.equal(results_flip['keypoints3d'],
                        original_keypoints3d).all()
    assert not np.equal(results_flip['keypoints2d'],
                        original_keypoints2d).all()
    assert not np.equal(results_flip['smplx_body_pose'],
                        original_body_pose).all()

    # test rotation
    results['rotation'] = 30
    results['scale'] = 0.25 * results['scale']
    transform = Rotation()
    results_rotated = transform(copy.deepcopy(results))
    assert not np.equal(results_rotated['img'].shape, original_img.shape).all()
    assert not np.equal(results_rotated['keypoints3d'],
                        original_keypoints3d).all()
    assert not np.equal(results_rotated['keypoints2d'],
                        original_keypoints2d).all()
    assert np.equal(results_rotated['smplx_body_pose'],
                    original_body_pose).all()
    assert results_rotated['rotation'] == 0.0
    assert results_rotated['ori_rotation'] == results['rotation']

    # test random affine
    transform = MeshAffine(img_res=224)
    results_affine = transform(copy.deepcopy(results_rotated))
    assert results_affine['img'].shape == (224, 224, 3)
    assert not np.equal(results_affine['img'].shape, original_img.shape).all()
    assert not np.equal(results_affine['keypoints3d'],
                        original_keypoints3d).all()
    assert not np.equal(results_affine['keypoints2d'],
                        original_keypoints2d).all()
    assert np.equal(results_affine['smplx_body_pose'],
                    original_body_pose).all()
    assert 'ori_img' in results_affine
    assert 'crop_transform' in results_affine

    # test random channel noise
    transform = RandomChannelNoise(noise_factor=0.4)
    results_noised = transform(copy.deepcopy(results_affine))
    assert not np.equal(results_noised['img'], results_affine['img']).all()
    assert not np.equal(results_noised['ori_img'],
                        results_affine['ori_img']).all()

    # test low resolution
    transform = SimulateLowRes(
        dist='categorical', cat_factors=(1.2, 1.5, 2.0, 3.0, 4.0, 8.0))
    results_lowres = transform(copy.deepcopy(results_affine))
    assert not np.equal(results_lowres['img'], results_affine['img']).all()
    assert np.equal(results_lowres['ori_img'], results_affine['ori_img']).all()

    # test bbox center jitter
    transform = BBoxCenterJitter(factor=1.0, dist='normal')
    results_jittered = transform(copy.deepcopy(results))
    assert not np.equal(results_jittered['center'], results['center']).all()
