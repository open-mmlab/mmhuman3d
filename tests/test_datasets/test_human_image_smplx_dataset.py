import numpy as np

from mmhuman3d.data.datasets import HumanImageSMPLXDataset


def test_human_image_smplx_dataset():
    train_dataset = HumanImageSMPLXDataset(
        data_prefix='tests/data',
        pipeline=[],
        dataset_name='EHF',
        ann_file='ehf_val.npz')
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
            model_path='data/body_models/smplx',
            joints_regressor='data/body_models/SMPLX_to_J14.npy'),
        ann_file='ehf_val.npz',
        face_vertex_ids_path='data/body_models/SMPL-X__FLAME_vertex_ids.npy',
        hand_vertex_ids_path='data/body_models/MANO_SMPLX_vertex_ids.pkl',
        convention='smplx')
    num_data = 100
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
