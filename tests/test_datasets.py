import os

from mmhuman3d.data.datasets import build_dataset


def check_dataset(dataset):
    for i, data in enumerate(dataset):
        assert data['bbox_xywh'].shape == (4, )
        assert data['keypoints2d'].shape == (144, 3)
        assert data['keypoints3d'].shape == (144, 4)

        assert data['smpl_body_pose'].shape == (23, 3)
        assert data['smpl_global_orient'].shape == (3, )
        assert data['smpl_betas'].shape == (10, )
        assert data['smpl_transl'].shape == (3, )
        assert isinstance(data['has_smpl'], float)

        assert data['smplx_body_pose'].shape == (21, 3)
        assert data['smplx_global_orient'].shape == (3, )
        assert data['smplx_betas'].shape == (10, )
        assert data['smplx_transl'].shape == (3, )
        assert data['smplx_left_hand_pose'].shape == (15, 3)
        assert data['smplx_right_hand_pose'].shape == (15, 3)
        assert data['smplx_expression'].shape == (10, )
        assert data['smplx_leye_pose'].shape == (3, )
        assert data['smplx_reye_pose'].shape == (3, )
        assert data['smplx_jaw_pose'].shape == (3, )
        assert isinstance(data['has_smplx'], float)

        assert data['mask'].shape == (144, )


def test_datasets():
    base_folder = './tests/data'
    assert os.path.exists(base_folder)

    cfg = {
        'type': None,
        'data_prefix': base_folder,
        'ann_file': 'sample_3dpw_train.npz',
        'pipeline': [dict(type='LoadImageFromFile')]
    }

    # test 3dpw
    cfg.update({'type': 'PW3D'})
    dataset = build_dataset(cfg)

    check_dataset(dataset)
