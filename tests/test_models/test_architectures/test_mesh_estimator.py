import torch

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.models.architectures.mesh_estimator import (
    ImageBodyModelEstimator,
    VideoBodyModelEstimator,
)
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.utils.geometry import project_points


def test_image_body_mesh_estimator():
    backbone = dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        norm_eval=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
    head = dict(type='HMRHead', feat_dim=2048)
    body_model_train = dict(
        type='SMPL',
        keypoint_src='smpl_45',
        keypoint_dst='smpl_45',
        model_path='data/body_models/smpl')
    body_model_test = dict(
        type='SMPL',
        keypoint_src='smpl_45',
        keypoint_dst='smpl_45',
        model_path='data/body_models/smpl')
    convention = 'smpl_45'
    loss_keypoints3d = dict(type='SmoothL1Loss', loss_weight=100)
    loss_keypoints2d = dict(type='SmoothL1Loss', loss_weight=10)
    loss_vertex = dict(type='L1Loss', loss_weight=2)
    loss_smpl_pose = dict(type='MSELoss', loss_weight=3)
    loss_smpl_betas = dict(type='MSELoss', loss_weight=0.02)
    loss_camera = dict(type='CameraPriorLoss', loss_weight=60)
    loss_adv = dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1)
    disc = dict(type='SMPLDiscriminator')
    model = ImageBodyModelEstimator()
    assert model.backbone is None
    assert model.neck is None
    assert model.head is None
    assert model.body_model_train is None
    assert model.body_model_test is None
    assert model.convention == 'human_data'
    assert model.loss_keypoints3d is None
    assert model.loss_keypoints2d is None
    assert model.loss_vertex is None
    assert model.loss_smpl_pose is None
    assert model.loss_smpl_betas is None
    assert model.loss_camera is None
    assert model.loss_adv is None
    assert model.disc is None

    model = ImageBodyModelEstimator(
        backbone=backbone,
        head=head,
        body_model_train=body_model_train,
        body_model_test=body_model_test,
        convention=convention,
        loss_keypoints3d=loss_keypoints3d,
        loss_keypoints2d=loss_keypoints2d,
        loss_vertex=loss_vertex,
        loss_smpl_pose=loss_smpl_pose,
        loss_smpl_betas=loss_smpl_betas,
        loss_camera=loss_camera,
        loss_adv=loss_adv,
        disc=disc)

    assert model.backbone is not None
    assert model.head is not None
    assert model.body_model_train is not None
    assert model.body_model_test is not None
    assert model.convention == 'smpl_45'
    assert model.loss_keypoints3d is not None
    assert model.loss_keypoints2d is not None
    assert model.loss_vertex is not None
    assert model.loss_smpl_pose is not None
    assert model.loss_smpl_betas is not None
    assert model.loss_camera is not None
    assert model.loss_adv is not None
    assert model.disc is not None


def test_video_body_mesh_estimator():
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
        type='SMPL',
        keypoint_src='smpl_45',
        keypoint_dst='smpl_45',
        model_path='data/body_models/smpl')
    body_model_test = dict(
        type='SMPL',
        keypoint_src='smpl_45',
        keypoint_dst='smpl_45',
        model_path='data/body_models/smpl')
    convention = 'smpl_45'
    loss_keypoints3d = dict(type='SmoothL1Loss', loss_weight=100)
    loss_keypoints2d = dict(type='SmoothL1Loss', loss_weight=10)
    loss_vertex = dict(type='L1Loss', loss_weight=2)
    loss_smpl_pose = dict(type='MSELoss', loss_weight=3)
    loss_smpl_betas = dict(type='MSELoss', loss_weight=0.02)
    loss_camera = dict(type='CameraPriorLoss', loss_weight=60)
    model = ImageBodyModelEstimator()
    assert model.backbone is None
    assert model.neck is None
    assert model.head is None
    assert model.body_model_train is None
    assert model.body_model_test is None
    assert model.registrant is None
    assert model.convention == 'human_data'
    assert model.loss_keypoints3d is None
    assert model.loss_keypoints2d is None
    assert model.loss_vertex is None
    assert model.loss_smpl_pose is None
    assert model.loss_smpl_betas is None
    assert model.loss_camera is None
    assert model.loss_adv is None
    assert model.disc is None

    model = VideoBodyModelEstimator(
        backbone=backbone,
        neck=neck,
        head=head,
        body_model_train=body_model_train,
        body_model_test=body_model_test,
        convention=convention,
        loss_keypoints3d=loss_keypoints3d,
        loss_keypoints2d=loss_keypoints2d,
        loss_vertex=loss_vertex,
        loss_smpl_pose=loss_smpl_pose,
        loss_smpl_betas=loss_smpl_betas,
        loss_camera=loss_camera)

    assert model.backbone is not None
    assert model.neck is not None
    assert model.head is not None
    assert model.body_model_train is not None
    assert model.body_model_test is not None
    assert model.convention == 'smpl_45'
    assert model.loss_keypoints3d is not None
    assert model.loss_keypoints2d is not None
    assert model.loss_vertex is not None
    assert model.loss_smpl_pose is not None
    assert model.loss_smpl_betas is not None
    assert model.loss_camera is not None


def test_compute_keypoints3d_loss():
    model = ImageBodyModelEstimator(
        convention='smpl_54',
        loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100))

    pred_keypoints3d = torch.zeros((32, 54, 3))
    gt_keypoints3d = torch.zeros((32, 54, 4))
    loss_empty = model.compute_keypoints3d_loss(pred_keypoints3d,
                                                gt_keypoints3d)
    assert loss_empty == 0

    pred_keypoints3d = torch.randn((32, 54, 3))
    gt_keypoints3d = torch.randn((32, 54, 4))
    gt_keypoints3d[:, :, 3] = torch.sigmoid(gt_keypoints3d[:, :, 3])
    loss = model.compute_keypoints3d_loss(pred_keypoints3d, gt_keypoints3d)
    assert loss > 0

    has_keypoints3d = torch.ones(32)
    loss = model.compute_keypoints3d_loss(
        pred_keypoints3d, gt_keypoints3d, has_keypoints3d=has_keypoints3d)
    assert loss > 0
    has_keypoints3d = torch.zeros(32)
    loss = model.compute_keypoints3d_loss(
        pred_keypoints3d, gt_keypoints3d, has_keypoints3d=has_keypoints3d)
    assert loss == 0


def test_compute_keypoints2d_loss():
    model = ImageBodyModelEstimator(
        convention='smpl_54',
        loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10))

    pred_keypoints3d = torch.zeros((32, 54, 3))
    gt_keypoints2d = torch.zeros((32, 54, 3))
    pred_cam = torch.randn((32, 3))
    loss_empty = model.compute_keypoints2d_loss(pred_keypoints3d, pred_cam,
                                                gt_keypoints2d)
    assert loss_empty == 0

    pred_keypoints3d = torch.randn((32, 54, 3))
    gt_keypoints2d = torch.randn((32, 54, 3))
    gt_keypoints2d[:, :, 2] = torch.sigmoid(gt_keypoints2d[:, :, 2])
    pred_cam = torch.randn((32, 3))
    loss = model.compute_keypoints2d_loss(pred_keypoints3d, pred_cam,
                                          gt_keypoints2d)
    assert loss > 0

    has_keypoints2d = torch.ones((32))
    loss = model.compute_keypoints2d_loss(
        pred_keypoints3d,
        pred_cam,
        gt_keypoints2d,
        has_keypoints2d=has_keypoints2d)
    assert loss > 0

    has_keypoints2d = torch.zeros((32))
    loss = model.compute_keypoints2d_loss(
        pred_keypoints3d,
        pred_cam,
        gt_keypoints2d,
        has_keypoints2d=has_keypoints2d)
    assert loss == 0


def test_compute_vertex_loss():
    model = ImageBodyModelEstimator(
        convention='smpl_54', loss_vertex=dict(type='L1Loss', loss_weight=2))

    pred_vertices = torch.randn((32, 4096, 3))
    gt_vertices = torch.randn((32, 4096, 3))
    has_smpl = torch.zeros((32))
    loss_empty = model.compute_vertex_loss(pred_vertices, gt_vertices,
                                           has_smpl)
    assert loss_empty == 0

    pred_vertices = torch.randn((32, 4096, 3))
    gt_vertices = torch.randn((32, 4096, 3))
    has_smpl = torch.ones((32))
    loss = model.compute_vertex_loss(pred_vertices, gt_vertices, has_smpl)
    assert loss > 0


def test_compute_smpl_pose_loss():
    model = ImageBodyModelEstimator(
        convention='smpl_54',
        loss_smpl_pose=dict(type='MSELoss', loss_weight=3))

    pred_rotmat = torch.randn((32, 24, 3, 3))
    gt_pose = torch.randn((32, 24, 3))
    has_smpl = torch.zeros((32))
    loss_empty = model.compute_smpl_pose_loss(pred_rotmat, gt_pose, has_smpl)
    assert loss_empty == 0

    pred_rotmat = torch.randn((32, 24, 3, 3))
    gt_pose = torch.randn((32, 24, 3))
    has_smpl = torch.ones((32))
    loss = model.compute_smpl_pose_loss(pred_rotmat, gt_pose, has_smpl)
    assert loss > 0


def test_compute_part_segm_loss():
    N = 1
    random_body_pose = torch.rand((N, 69))
    body_model_train = dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_49',
        model_path='data/body_models/smpl',
        extra_joints_regressor='data/body_models/J_regressor_extra.npy')
    body_model = build_body_model(body_model_train)

    body_model_output = body_model(body_pose=random_body_pose, )
    gt_model_joins = body_model_output['joints'].detach()
    cam = torch.ones(N, 3)
    gt_keypoints2d = project_points(
        gt_model_joins, cam, focal_length=5000, img_res=224)
    loss_segm_mask = dict(type='CrossEntropyLoss', loss_weight=60)

    gt_keypoints2d = torch.cat([gt_keypoints2d, torch.ones(N, 49, 1)], dim=-1)
    model = ImageBodyModelEstimator(
        body_model_train=body_model_train,
        loss_segm_mask=loss_segm_mask,
    )
    gt_vertices = torch.randn(N, 6890, 3)
    pred_heatmap = torch.zeros(N, 25, 224, 224)
    pred_heatmap[:, 0, :, :] = 1
    has_smpl = torch.ones((N))

    loss = model.compute_part_segmentation_loss(
        pred_heatmap,
        gt_vertices,
        has_smpl=has_smpl,
        gt_keypoints2d=gt_keypoints2d,
        gt_model_joints=gt_model_joins)
    assert loss > 0


def test_compute_smpl_betas_loss():
    model = ImageBodyModelEstimator(
        convention='smpl_54',
        loss_smpl_betas=dict(type='MSELoss', loss_weight=0.02))

    pred_betas = torch.randn((32, 10))
    gt_betas = torch.randn((32, 10))
    has_smpl = torch.zeros((32))
    loss_empty = model.compute_smpl_betas_loss(pred_betas, gt_betas, has_smpl)
    assert loss_empty == 0

    pred_betas = torch.randn((32, 10))
    gt_betas = torch.randn((32, 10))
    has_smpl = torch.ones((32))
    loss = model.compute_smpl_betas_loss(pred_betas, gt_betas, has_smpl)
    assert loss > 0


def test_compute_camera_loss():
    model = ImageBodyModelEstimator(
        convention='smpl_54',
        loss_camera=dict(type='CameraPriorLoss', loss_weight=60),
    )

    pred_cameras = torch.randn((32, 3))
    loss = model.compute_camera_loss(pred_cameras)
    assert loss > 0


def test_compute_losses():
    N = 32
    predictions = {}
    predictions['pred_shape'] = torch.randn(N, 10)
    predictions['pred_pose'] = torch.randn(N, 24, 3, 3)
    predictions['pred_cam'] = torch.randn(N, 3)

    targets = {}
    targets['keypoints3d'] = torch.randn(N, 45, 4)
    targets['keypoints2d'] = torch.randn(N, 45, 3)
    targets['has_smpl'] = torch.ones(N)
    targets['smpl_body_pose'] = torch.randn(N, 23, 3)
    targets['smpl_global_orient'] = torch.randn(N, 3)
    targets['smpl_betas'] = torch.randn(N, 10)

    model = ImageBodyModelEstimator(convention='smpl_54')
    loss = model.compute_losses(predictions, targets)
    assert loss == {}

    model = ImageBodyModelEstimator(
        convention='smpl_45',
        body_model_train=dict(
            type='SMPL',
            keypoint_src='smpl_45',
            keypoint_dst='smpl_45',
            model_path='data/body_models/smpl'),
        loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
        loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
        loss_vertex=dict(type='L1Loss', loss_weight=2),
        loss_smpl_pose=dict(type='MSELoss', loss_weight=3),
        loss_smpl_betas=dict(type='MSELoss', loss_weight=0.02),
        loss_camera=dict(type='CameraPriorLoss', loss_weight=60))

    loss = model.compute_losses(predictions, targets)
    assert 'keypoints3d_loss' in loss
    assert 'keypoints2d_loss' in loss
    assert 'vertex_loss' in loss
    assert 'smpl_pose_loss' in loss
    assert 'smpl_betas_loss' in loss
    assert 'camera_loss' in loss


def test_run_registration():
    batch_size = 2
    body_model = dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_49',
        keypoint_approximate=True,
        model_path='data/body_models/smpl',
        extra_joints_regressor='data/body_models/J_regressor_extra.npy',
        batch_size=batch_size)

    camera = build_cameras(
        dict(
            type='PerspectiveCameras',
            convention='opencv',
            in_ndc=False,
            focal_length=5000,
            image_size=(224, 224),
            principal_point=(112, 112)))

    registrant = dict(
        type='SMPLify',
        body_model=body_model,
        num_epochs=1,
        stages=[
            dict(
                num_iter=1,
                fit_global_orient=True,
                fit_transl=True,
                fit_body_pose=False,
                fit_betas=False)
        ],
        optimizer=dict(type='Adam', lr=1e-2, betas=(0.9, 0.999)),
        keypoints2d_loss=dict(
            type='KeypointMSELoss',
            loss_weight=1.0,
            reduction='sum',
            sigma=100),
        device=torch.device('cpu'),
        camera=camera)

    registration = dict(mode='in_the_loop', registrant=registrant)

    model = ImageBodyModelEstimator(
        body_model_train=body_model, registration=registration)
    assert model.registrant is not None
    assert model.fits_dict is not None

    transl = torch.Tensor([0, 0, 1]).view(1, 3).expand(batch_size, -1)

    predictions = dict(
        pred_pose=torch.zeros((batch_size, 24, 3, 3)),
        pred_shape=torch.zeros((batch_size, 10)),
        pred_cam=transl,
    )

    # generate 2D keypoints
    smpl = build_body_model(body_model)
    keypoints3d = smpl(transl=transl)['joints'].detach()
    keypoints2d_xyd = camera.transform_points_screen(keypoints3d)
    keypoints2d = keypoints2d_xyd[..., :2]
    keypoints2d_conf = torch.ones(*keypoints2d.shape[:2], 1)
    keypoints2d = torch.cat([keypoints2d, keypoints2d_conf], dim=-1)

    targets = dict(
        img_metas=[dict(dataset_name='coco'),
                   dict(dataset_name='h36m')],
        sample_idx=torch.zeros((batch_size, 1), dtype=torch.int),
        is_flipped=torch.tensor([0, 1], dtype=torch.int),
        rotation=torch.tensor([0.0, 0.1]),
        smpl_betas=torch.zeros((batch_size, 10)),
        smpl_global_orient=torch.zeros((batch_size, 3)),
        smpl_body_pose=torch.zeros((batch_size, 69)),
        keypoints2d=keypoints2d,
        has_smpl=torch.tensor([0, 1], dtype=torch.int))

    model.run_registration(predictions=predictions, targets=targets)
    assert 'valid_fit' in targets
    assert 'opt_vertices' in targets
    assert 'opt_cam_t' in targets
    assert 'opt_joints' in targets
    assert 'opt_pose' in targets
    assert 'opt_betas' in targets
