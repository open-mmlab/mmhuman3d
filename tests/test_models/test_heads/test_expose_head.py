import numpy as np
import torch

from mmhuman3d.models.heads.builder import ExPoseBodyHead, ExPoseFaceHead, ExPoseHandHead

def test_expose_body_head():
    head_cfg =  dict(
        num_betas = 10,
        num_expression_coeffs = 10,
        mean_pose_path = 'data/body_models/all_means.pkl',
        shape_mean_path = 'data/body_models/shape_mean.npy',
        pose_param_conf = [
            dict(
                name = 'global_orient',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = True),
            dict(
                name = 'body_pose',
                num_angles = 21,
                use_mean = True,
                rotate_axis_x = False),
            dict(
                name = 'left_hand_pose',
                num_angles = 15,
                use_mean = True,
                rotate_axis_x = False),
            dict(
                name = 'right_hand_pose',
                num_angles = 15,
                use_mean = True,
                rotate_axis_x = False),
            dict(
                name = 'jaw_pose',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = False),
        ],
        input_feat_dim = 2048,
        regressor_cfg = dict(
            layers = [1024,1024],
            activ_type = 'none',
            dropout = 0.5,
            gain = 0.01
        ),
        camera_cfg = dict(
            pos_func = 'softplus',
            mean_scale = 0.9
        ),
    )
    head = ExPoseBodyHead(**head_cfg)

    batch_size = 2
    features = np.random.rand(batch_size,2048)
    features = torch.tensor(features).float()
    predictions = head(features)
    pred_keys = ['pred_param','pred_cam','pred_raw']
    param_keys = ['global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','betas','expression']
    raw_keys = ['raw_global_orient','raw_body_pose','raw_left_hand_pose','raw_right_hand_pose','raw_jaw_pose']
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

def test_expose_hand_head():
    head_cfg =  dict(
        num_betas = 10,
        mean_pose_path = 'data/body_models/all_means.pkl',
        pose_param_conf = [
            dict(
                name = 'global_orient',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = False),
            dict(
                name = 'right_hand_pose',
                num_angles = 15,
                use_mean = True,
                rotate_axis_x = False),
        ],
        input_feat_dim = 512,
        regressor_cfg = dict(
            layers = [1024,1024],
            activ_type = 'ReLU',
            dropout = 0.5,
            gain = 0.01
        ),
        camera_cfg = dict(
            pos_func = 'softplus',
            mean_scale = 0.9
        ),
    )
    head = ExPoseHandHead(**head_cfg)

    batch_size = 2
    features = np.random.rand(batch_size,512,7,7)
    features = [torch.tensor(features).float()]
    predictions = head(features)
    pred_keys = ['pred_param','pred_cam','pred_raw']
    param_keys = ['global_orient','right_hand_pose','betas']
    raw_keys = ['raw_global_orient','raw_right_hand_pose']
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
    head_cfg =  dict(
        num_betas = 100,
        num_expression_coeffs = 50,
        mean_pose_path = 'data/body_models/all_means.pkl',
        pose_param_conf = [
            dict(
                name = 'global_orient',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = True),
            dict(
                name = 'jaw_pose',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = False),
        ],
        input_feat_dim = 512,
        regressor_cfg = dict(
            layers = [1024,1024],
            activ_type = 'ReLU',
            dropout = 0.5,
            gain = 0.01
        ),
        camera_cfg = dict(
            pos_func = 'softplus',
            mean_scale = 8.0
        ),
    )
    head = ExPoseFaceHead(**head_cfg)

    batch_size = 2
    features = np.random.rand(batch_size,512,8,8)
    features = [torch.tensor(features).float()]
    predictions = head(features)
    pred_keys = ['pred_param','pred_cam','pred_raw']
    param_keys = ['global_orient','jaw_pose','betas','expression']
    raw_keys = ['raw_global_orient','raw_jaw_pose']
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