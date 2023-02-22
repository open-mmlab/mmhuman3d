_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'])

img_res = 224

body_model = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst='smpl_49',
    keypoint_approximate=True,
    model_path='data/body_models/smpl',
    extra_joints_regressor='data/body_models/J_regressor_extra.npy')

registrant = dict(
    type='SMPLify',
    body_model=body_model,
    num_epochs=1,
    stages=[
        # stage 1
        dict(
            num_iter=50,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False),
        # stage 2
        dict(
            num_iter=50,
            fit_global_orient=True,
            fit_transl=False,
            fit_body_pose=True,
            fit_betas=True),
    ],
    optimizer=dict(type='Adam', lr=1e-2, betas=(0.9, 0.999)),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100),
    shape_prior_loss=dict(
        type='ShapePriorLoss', loss_weight=5.0**2, reduction='sum'),
    joint_prior_loss=dict(
        type='JointPriorLoss', loss_weight=15.2**2, reduction='sum'),
    pose_prior_loss=dict(
        type='MaxMixturePrior',
        prior_folder='data',
        num_gaussians=8,
        loss_weight=4.78**2,
        reduction='sum'),
    ignore_keypoints=[
        'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
        'right_hip_extra', 'left_hip_extra'
    ],
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(img_res, img_res),
        principal_point=(img_res / 2, img_res / 2)))

registration = dict(mode='in_the_loop', registrant=registrant)

# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=3e-5), head=dict(type='Adam', lr=3e-5))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.1, step=[3])
runner = dict(type='EpochBasedRunner', max_epochs=10)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# model settings
model = dict(
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        norm_eval=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    head=dict(
        type='HMRHead',
        feat_dim=2048,
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model_train=body_model,
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    registration=registration,
    convention='smpl_49',
    loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_vertex=dict(type='L1Loss', loss_weight=2),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=3),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.02),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=60),
    init_cfg=dict(
        type='Pretrained', checkpoint='data/pretrained/spin_pretrain.pth'))
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'is_flipped', 'center',
    'scale', 'rotation', 'sample_idx'
]
meta_data_keys = ['dataset_name', 'image_path']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_49'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys)
]
data_keys.remove('is_flipped')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys)
]

inference_pipeline = [
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='h36m_mosh_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_mpi_inf_3dhp_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='lsp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_lsp_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='lspet',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_lspet_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpii',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_mpii_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_coco_2014_train.npz'),
        ],
        partition=[0.35, 0.15, 0.1, 0.10, 0.10, 0.2],
        num_data=100000,
    ),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='spin_pw3d_test.npz'),
)
