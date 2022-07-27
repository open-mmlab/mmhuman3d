import torch

from mmhuman3d.models.architectures.expressive_mesh_estimator import (
    SMPLXImageBodyModelEstimator,
    pose2rotmat,
)


def test_smplx_image_body_mesh_estimator():
    model = SMPLXImageBodyModelEstimator()
    assert model.backbone is None
    assert model.neck is None
    assert model.head is None
    assert model.body_model_train is None
    assert model.body_model_test is None
    assert model.convention == 'human_data'
    assert model.apply_face_model is False
    assert model.apply_hand_model is False
    assert model.loss_keypoints3d is None
    assert model.loss_keypoints2d is None
    assert model.loss_smplx_betas is None
    assert model.loss_smplx_betas_piror is None
    assert model.loss_smplx_body_pose is None
    assert model.loss_smplx_jaw_pose is None
    assert model.loss_smplx_expression is None
    assert model.loss_smplx_global_orient is None
    assert model.loss_smplx_hand_pose is None
    assert model.loss_camera is None

    backbone = dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        norm_eval=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
    neck = dict(type='TemporalGRUEncoder', num_layers=2, hidden_size=1024)
    head = dict(type='HMRHead', feat_dim=2048)
    body_model_train = dict(
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
    body_model_test = dict(
        type='SMPLXLayer',
        num_expression_coeffs=10,
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=True,
        model_path='data/body_models/smplx',
        keypoint_src='lsp',
        keypoint_dst='lsp')
    loss_keypoints3d = dict(type='L1Loss', reduction='sum', loss_weight=1)
    loss_keypoints2d = dict(type='L1Loss', reduction='sum', loss_weight=1)
    loss_smplx_global_orient = dict(
        type='RotationDistance', reduction='sum', loss_weight=1)
    loss_smplx_body_pose = dict(
        type='RotationDistance', reduction='sum', loss_weight=1)
    loss_smplx_jaw_pose = dict(
        type='RotationDistance', reduction='sum', loss_weight=1)
    loss_smplx_hand_pose = dict(
        type='RotationDistance', reduction='sum', loss_weight=1)
    loss_smplx_betas = dict(type='MSELoss', reduction='sum', loss_weight=0.001)
    loss_smplx_expression = dict(
        type='MSELoss', reduction='sum', loss_weight=1)
    loss_smplx_betas_prior = dict(
        type='ShapeThresholdPriorLoss', margin=3.0, norm='l2', loss_weight=1)
    convention = 'smplx'

    model = SMPLXImageBodyModelEstimator(
        backbone=backbone,
        neck=neck,
        head=head,
        body_model_train=body_model_train,
        body_model_test=body_model_test,
        convention=convention,
        loss_keypoints2d=loss_keypoints2d,
        loss_keypoints3d=loss_keypoints3d,
        loss_smplx_betas=loss_smplx_betas,
        loss_smplx_betas_prior=loss_smplx_betas_prior,
        loss_smplx_body_pose=loss_smplx_body_pose,
        loss_smplx_expression=loss_smplx_expression,
        loss_smplx_global_orient=loss_smplx_global_orient,
        loss_smplx_hand_pose=loss_smplx_hand_pose,
        loss_smplx_jaw_pose=loss_smplx_jaw_pose)
    assert model.backbone is not None
    assert model.neck is not None
    assert model.head is not None
    assert model.body_model_train is not None
    assert model.body_model_test is not None
    assert model.convention == 'smplx'
    assert model.loss_keypoints3d is not None
    assert model.loss_keypoints2d is not None
    assert model.loss_smplx_betas is not None
    assert model.loss_smplx_betas_piror is not None
    assert model.loss_smplx_body_pose is not None
    assert model.loss_smplx_jaw_pose is not None
    assert model.loss_smplx_expression is not None
    assert model.loss_smplx_global_orient is not None
    assert model.loss_smplx_hand_pose is not None

    model = SMPLXImageBodyModelEstimator(
        backbone=backbone,
        neck=neck,
        head=head,
        body_model_train=body_model_train,
        body_model_test=body_model_test,
        convention=convention,
        extra_hand_model_cfg=dict(
            backbone=backbone,
            neck=neck,
            head=head,
            crop_cfg=dict(
                img_res=256,
                scale_factor=3.0,
                crop_size=224,
            ),
            loss_hand_crop=dict(type='L1Loss', reduction='sum',
                                loss_weight=1)),
        extra_face_model_cfg=dict(
            backbone=backbone,
            neck=neck,
            head=head,
            crop_cfg=dict(
                img_res=256,
                scale_factor=2.0,
                crop_size=256,
            ),
            loss_face_crop=dict(type='L1Loss', reduction='sum', loss_weight=1),
        ),
        frozen_batchnorm=True)
    assert model.apply_hand_model is True
    assert model.hand_backbone is not None
    assert model.hand_neck is not None
    assert model.hand_head is not None
    assert model.crop_hand_func is not None
    assert model.hand_merge_func is not None
    assert model.hand_crop_loss is not None
    assert model.apply_face_model is True
    assert model.face_backbone is not None
    assert model.face_neck is not None
    assert model.face_head is not None
    assert model.crop_face_func is not None
    assert model.face_merge_func is not None
    assert model.face_crop_loss is not None


def test_compute_keypoints3d_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_keypoints3d=dict(type='L1Loss', reduction='sum', loss_weight=1))
    pred_keypoints3d = torch.zeros((32, 144, 3))
    gt_keypoints3d = torch.zeros((32, 144, 4))
    gt_keypoints3d[:, :, -1] = 1
    loss_empty = model.compute_keypoints3d_loss(pred_keypoints3d,
                                                gt_keypoints3d)
    assert loss_empty == 0


def test_compute_keypoints2d_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_keypoints2d=dict(type='L1Loss', reduction='sum', loss_weight=1))

    pred_keypoints3d = torch.zeros((32, 144, 3))
    gt_keypoints2d = torch.zeros((32, 144, 3))
    pred_cam = torch.zeros((32, 3))
    loss_empty = model.compute_keypoints2d_loss(pred_keypoints3d, pred_cam,
                                                gt_keypoints2d)
    assert loss_empty == 0


def test_smplx_body_pose_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_smplx_body_pose=dict(
            type='L1Loss', reduction='sum', loss_weight=1))

    pred_rotmat = torch.eye(3).expand((32, 21, 3, 3))
    gt_pose = torch.zeros((32, 21, 3))
    _ = pose2rotmat(gt_pose)
    has_smplx_body_pose = torch.ones((32))
    loss_empty = model.compute_smplx_body_pose_loss(pred_rotmat, gt_pose,
                                                    has_smplx_body_pose)

    assert loss_empty == 0


def test_smplx_global_orient_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_smplx_global_orient=dict(
            type='L1Loss', reduction='sum', loss_weight=1))

    pred_rotmat = torch.eye(3).expand((32, 1, 3, 3))
    gt_global_orient = torch.zeros((32, 1, 3))
    has_smplx_global_orient = torch.ones((32))
    loss_empty = model.compute_smplx_global_orient_loss(
        pred_rotmat, gt_global_orient, has_smplx_global_orient)

    assert loss_empty == 0


def test_smplx_jaw_pose_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_smplx_jaw_pose=dict(
            type='L1Loss', reduction='sum', loss_weight=1))

    pred_rotmat = torch.eye(3).expand((32, 1, 3, 3))
    gt_jaw_pose = torch.zeros((32, 1, 3))
    has_smplx_jaw_pose = torch.ones((32))
    face_conf = torch.rand((32, 10))
    loss_empty = model.compute_smplx_jaw_pose_loss(pred_rotmat, gt_jaw_pose,
                                                   has_smplx_jaw_pose,
                                                   face_conf)

    assert loss_empty == 0


def test_smplx_hand_pose_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_smplx_hand_pose=dict(
            type='L1Loss', reduction='sum', loss_weight=1))

    pred_rotmat = torch.eye(3).expand((32, 15, 3, 3))
    gt_hand_pose = torch.zeros((32, 15, 3))
    has_smplx_hand_pose = torch.ones((32))
    hand_conf = torch.rand((32, 10))
    loss_empty = model.compute_smplx_hand_pose_loss(pred_rotmat, gt_hand_pose,
                                                    has_smplx_hand_pose,
                                                    hand_conf)

    assert loss_empty == 0


def test_smplx_betas_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_smplx_betas=dict(type='L1Loss', reduction='sum', loss_weight=1))

    pred_betas = torch.zeros((32, 10))
    gt_betas = torch.zeros((32, 10))
    has_smplx_betas = torch.ones((32))
    loss_empty = model.compute_smplx_betas_loss(pred_betas, gt_betas,
                                                has_smplx_betas)

    assert loss_empty == 0


def test_smplx_expression_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_smplx_expression=dict(
            type='L1Loss', reduction='sum', loss_weight=1))

    pred_expression = torch.zeros((32, 10))
    gt_expression = torch.zeros((32, 10))
    has_smplx_expression = torch.ones((32))
    face_conf = torch.rand((32, 10))
    loss_empty = model.compute_smplx_expression_loss(pred_expression,
                                                     gt_expression,
                                                     has_smplx_expression,
                                                     face_conf)

    assert loss_empty == 0


def test_smplx_betas_prior_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        loss_smplx_betas_prior=dict(
            type='ShapeThresholdPriorLoss',
            margin=3.0,
            norm='l2',
            loss_weight=1))

    pred_betas = torch.zeros((32, 10))
    loss_empty = model.compute_smplx_betas_prior_loss(pred_betas)

    assert loss_empty == 0


def test_hand_crop_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        extra_hand_model_cfg=dict(
            loss_hand_crop=dict(type='L1Loss', reduction='sum',
                                loss_weight=1)),
    )
    pred_keypoints3d = torch.zeros((32, 144, 3))
    pred_cam = torch.randn((32, 3))
    gt_keypoints2d = torch.zeros((32, 144, 3))
    hand_crop_info = dict(
        hand_inv_crop_transforms=torch.randn((32, 3, 3)),
        left_hand_crop_transform=torch.randn((32, 3, 3)),
        right_hand_crop_transform=torch.randn((32, 3, 3)))
    loss_empty = model.compute_hand_crop_loss(pred_keypoints3d, pred_cam,
                                              gt_keypoints2d, hand_crop_info)
    assert loss_empty == 0


def test_face_crop_loss():
    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        extra_face_model_cfg=dict(
            loss_face_crop=dict(type='L1Loss', reduction='sum',
                                loss_weight=1)),
    )
    pred_keypoints3d = torch.zeros((32, 144, 3))
    pred_cam = torch.randn((32, 3))
    gt_keypoints2d = torch.zeros((32, 144, 3))
    face_crop_info = dict(
        face_inv_crop_transforms=torch.randn((32, 3, 3)),
        face_crop_transform=torch.randn((32, 3, 3)),
    )
    loss_empty = model.compute_face_crop_loss(pred_keypoints3d, pred_cam,
                                              gt_keypoints2d, face_crop_info)
    assert loss_empty == 0


def test_compute_losses():
    N = 32
    predictions = {}
    predictions['pred_param'] = dict(
        global_orient=torch.randn(N, 1, 3, 3),
        body_pose=torch.randn(N, 21, 3, 3),
        jaw_pose=torch.randn(N, 1, 3, 3),
        right_hand_pose=torch.randn(N, 15, 3, 3),
        left_hand_pose=torch.randn(N, 15, 3, 3),
        betas=torch.randn(N, 10),
        expression=torch.randn(N, 10))
    predictions['pred_cam'] = torch.randn(N, 3)
    predictions['hand_crop_info'] = dict(
        hand_inv_crop_transforms=torch.randn((N, 3, 3)),
        left_hand_crop_transform=torch.randn((N, 3, 3)),
        right_hand_crop_transform=torch.randn((N, 3, 3)))
    predictions['face_crop_info'] = dict(
        face_inv_crop_transforms=torch.randn((N, 3, 3)),
        face_crop_transform=torch.randn((N, 3, 3)),
    )

    targets = {}
    targets['keypoints3d'] = torch.randn(N, 144, 4)
    targets['has_keypoints3d'] = torch.ones(N)
    targets['keypoints2d'] = torch.randn(N, 144, 3)
    targets['has_keypoints2d'] = torch.ones(N)
    targets['img'] = torch.randn(N, 3, 256, 256)
    targets['smplx_global_orient'] = torch.randn(N, 1, 3)
    targets['has_smplx_global_orient'] = torch.ones(N)
    targets['smplx_body_pose'] = torch.randn(N, 21, 3)
    targets['has_smplx_body_pose'] = torch.ones(N)
    targets['smplx_jaw_pose'] = torch.randn(N, 1, 3)
    targets['has_smplx_jaw_pose'] = torch.ones(N)
    targets['smplx_right_hand_pose'] = torch.randn(N, 15, 3)
    targets['has_smplx_right_hand_pose'] = torch.ones(N)
    targets['smplx_left_hand_pose'] = torch.randn(N, 15, 3)
    targets['has_smplx_left_hand_pose'] = torch.ones(N)
    targets['smplx_betas'] = torch.randn(N, 10)
    targets['has_smplx_betas'] = torch.ones(N)
    targets['smplx_expression'] = torch.randn(N, 10)
    targets['has_smplx_expression'] = torch.ones(N)

    model = SMPLXImageBodyModelEstimator(convention='smplx')
    loss = model.compute_losses(predictions, targets)
    assert loss == {}

    model = SMPLXImageBodyModelEstimator(
        convention='smplx',
        body_model_train=dict(
            type='SMPLXLayer',
            num_expression_coeffs=10,
            num_betas=10,
            use_face_contour=True,
            use_pca=False,
            flat_hand_mean=True,
            model_path='data/body_models/smplx',
            keypoint_src='smplx',
            keypoint_dst='smplx',
        ),
        loss_keypoints3d=dict(type='L1Loss', reduction='sum', loss_weight=1),
        loss_keypoints2d=dict(type='L1Loss', reduction='sum', loss_weight=1),
        loss_smplx_body_pose=dict(
            type='L1Loss', reduction='sum', loss_weight=1),
        loss_smplx_global_orient=dict(
            type='L1Loss', reduction='sum', loss_weight=1),
        loss_smplx_jaw_pose=dict(
            type='L1Loss', reduction='sum', loss_weight=1),
        loss_smplx_hand_pose=dict(
            type='L1Loss', reduction='sum', loss_weight=1),
        loss_smplx_betas=dict(type='L1Loss', reduction='sum', loss_weight=1),
        loss_smplx_expression=dict(
            type='L1Loss', reduction='sum', loss_weight=1),
        loss_smplx_betas_prior=dict(
            type='ShapeThresholdPriorLoss',
            margin=3.0,
            norm='l2',
            loss_weight=1),
        extra_hand_model_cfg=dict(
            loss_hand_crop=dict(type='L1Loss', reduction='sum',
                                loss_weight=1)),
        extra_face_model_cfg=dict(
            loss_face_crop=dict(type='L1Loss', reduction='sum',
                                loss_weight=1)),
    )
    loss = model.compute_losses(predictions, targets)
    assert 'keypoints3d_loss' in loss
    assert 'keypoints2d_loss' in loss
    assert 'smplx_global_orient_loss' in loss
    assert 'smplx_body_pose_loss' in loss
    assert 'smplx_jaw_pose_loss' in loss
    assert 'smplx_right_hand_pose_loss' in loss
    assert 'smplx_left_hand_pose_loss' in loss
    assert 'smplx_betas_loss' in loss
    assert 'smplx_expression_loss' in loss
    assert 'smplx_betas_prior_loss' in loss
    assert 'hand_crop_loss' in loss
    assert 'face_crop_loss' in loss
