import numpy as np
import torch

from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.heads.builder import (
    ExPoseBodyHead,
    ExPoseFaceHead,
    ExPoseHandHead,
)
from mmhuman3d.models.utils import (
    SMPLXFaceCropFunc,
    SMPLXFaceMergeFunc,
    SMPLXHandCropFunc,
    SMPLXHandMergeFunc,
)

body_model_cfg = dict(
    type='SMPLXLayer',
    num_expression_coeffs=10,
    num_betas=10,
    use_face_contour=True,
    use_pca=False,
    flat_hand_mean=True,
    model_path='data/body_models/smplx',
    keypoint_src='smplx',
    keypoint_dst='smplx',
)
body_model = build_body_model(body_model_cfg)


def test_expose_body_head():
    head_cfg = dict(
        num_betas=10,
        num_expression_coeffs=10,
        mean_pose_path='data/body_models/all_means.pkl',
        shape_mean_path='data/body_models/shape_mean.npy',
        pose_param_conf=[
            dict(
                name='global_orient',
                num_angles=1,
                use_mean=False,
                rotate_axis_x=True),
            dict(
                name='body_pose',
                num_angles=21,
                use_mean=True,
                rotate_axis_x=False),
            dict(
                name='left_hand_pose',
                num_angles=15,
                use_mean=True,
                rotate_axis_x=False),
            dict(
                name='right_hand_pose',
                num_angles=15,
                use_mean=True,
                rotate_axis_x=False),
            dict(
                name='jaw_pose',
                num_angles=1,
                use_mean=False,
                rotate_axis_x=False),
        ],
        input_feat_dim=2048,
        regressor_cfg=dict(
            layers=[1024, 1024], activ_type='none', dropout=0.5, gain=0.01),
        camera_cfg=dict(pos_func='softplus', mean_scale=0.9),
    )
    head = ExPoseBodyHead(**head_cfg)

    batch_size = 2
    features = np.random.rand(batch_size, 2048)
    features = torch.tensor(features).float()
    predictions = head(features)
    pred_keys = ['pred_param', 'pred_cam', 'pred_raw']
    param_keys = [
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'betas', 'expression'
    ]
    raw_keys = [
        'raw_global_orient', 'raw_body_pose', 'raw_left_hand_pose',
        'raw_right_hand_pose', 'raw_jaw_pose'
    ]
    for k in pred_keys:
        assert k in predictions
        if k == 'pred_param':
            for pk in param_keys:
                assert pk in predictions[k]
                assert predictions[k][pk].shape[0] == batch_size
        elif k == 'pred_raw':
            for rk in raw_keys:
                assert rk in predictions[k]
                assert predictions[k][rk].shape[0] == batch_size
        else:
            assert predictions[k].shape[0] == batch_size

    img_meta = dict(
        ori_img=torch.rand(3, 300, 300),
        crop_transform=np.random.rand(2, 3),
        scale=np.random.rand(2))
    img_metas = [img_meta, img_meta]
    face_crop = SMPLXFaceCropFunc(head, body_model)
    face_imgs, face_mean, crop_info = face_crop(predictions, img_metas)
    assert face_imgs.shape[0] == batch_size
    hand_crop = SMPLXHandCropFunc(head, body_model)
    hand_imgs, hand_mean, crop_info = hand_crop(predictions, img_metas)
    assert hand_imgs.shape[0] == 2 * batch_size

    hand_predictions = dict(
        pred_param=dict(
            global_orient=torch.rand(batch_size * 2, 1, 3, 3),
            right_hand_pose=torch.rand(batch_size * 2, 15, 3, 3)))
    hand_merge = SMPLXHandMergeFunc(body_model)
    predictions = hand_merge(predictions, hand_predictions)

    face_predictions = dict(
        pred_param=dict(
            global_orient=torch.rand(batch_size, 1, 3, 3),
            jaw_pose=torch.rand(batch_size, 1, 3, 3),
            expression=torch.rand(batch_size, 10)))
    face_merge = SMPLXFaceMergeFunc(body_model)
    predictions = face_merge(predictions, face_predictions)
    assert predictions['pred_param']['jaw_pose'].shape[0] == batch_size
    assert predictions['pred_param']['right_hand_pose'].shape[0] == batch_size
    assert predictions['pred_param']['left_hand_pose'].shape[0] == batch_size
    assert predictions['pred_param']['expression'].shape[0] == batch_size


def test_expose_hand_head():
    head_cfg = dict(
        num_betas=10,
        mean_pose_path='data/body_models/all_means.pkl',
        pose_param_conf=[
            dict(
                name='global_orient',
                num_angles=1,
                use_mean=False,
                rotate_axis_x=False),
            dict(
                name='right_hand_pose',
                num_angles=15,
                use_mean=True,
                rotate_axis_x=False),
        ],
        input_feat_dim=512,
        regressor_cfg=dict(
            layers=[1024, 1024], activ_type='ReLU', dropout=0.5, gain=0.01),
        camera_cfg=dict(pos_func='softplus', mean_scale=0.9),
    )
    head = ExPoseHandHead(**head_cfg)

    batch_size = 2
    features = np.random.rand(batch_size, 512, 7, 7)
    features = [torch.tensor(features).float()]
    predictions = head(features)
    pred_keys = ['pred_param', 'pred_cam', 'pred_raw']
    param_keys = ['global_orient', 'right_hand_pose', 'betas']
    raw_keys = ['raw_global_orient', 'raw_right_hand_pose']
    for k in pred_keys:
        assert k in predictions
        if k == 'pred_param':
            for pk in param_keys:
                assert pk in predictions[k]
                assert predictions[k][pk].shape[0] == batch_size
        elif k == 'pred_raw':
            for rk in raw_keys:
                assert rk in predictions[k]
                assert predictions[k][rk].shape[0] == batch_size
        else:
            assert predictions[k].shape[0] == batch_size


def test_expose_face_head():
    head_cfg = dict(
        num_betas=100,
        num_expression_coeffs=50,
        mean_pose_path='data/body_models/all_means.pkl',
        pose_param_conf=[
            dict(
                name='global_orient',
                num_angles=1,
                use_mean=False,
                rotate_axis_x=True),
            dict(
                name='jaw_pose',
                num_angles=1,
                use_mean=False,
                rotate_axis_x=False),
        ],
        input_feat_dim=512,
        regressor_cfg=dict(
            layers=[1024, 1024], activ_type='ReLU', dropout=0.5, gain=0.01),
        camera_cfg=dict(pos_func='softplus', mean_scale=8.0),
    )
    head = ExPoseFaceHead(**head_cfg)

    batch_size = 2
    features = np.random.rand(batch_size, 512, 8, 8)
    features = [torch.tensor(features).float()]
    predictions = head(features)
    pred_keys = ['pred_param', 'pred_cam', 'pred_raw']
    param_keys = ['global_orient', 'jaw_pose', 'betas', 'expression']
    raw_keys = ['raw_global_orient', 'raw_jaw_pose']
    for k in pred_keys:
        assert k in predictions
        if k == 'pred_param':
            for pk in param_keys:
                assert pk in predictions[k]
                assert predictions[k][pk].shape[0] == batch_size
        elif k == 'pred_raw':
            for rk in raw_keys:
                assert rk in predictions[k]
                assert predictions[k][rk].shape[0] == batch_size
        else:
            assert predictions[k].shape[0] == batch_size
