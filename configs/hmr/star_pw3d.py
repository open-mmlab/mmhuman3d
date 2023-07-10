_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'])
# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=2.5e-4), head=dict(type='Adam', lr=2.5e-4))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=100)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

img_res = 224

# model settings
model = dict(
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        norm_eval=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(type='HMRHead', feat_dim=2048),
    body_model_train=dict(
        type='STAR',
        keypoint_src='smpl',
        keypoint_dst='star',
        model_path='data/body_models/star',
        keypoint_approximate=True),
    body_model_test=dict(
        type='STAR',
        keypoint_src='smpl',
        keypoint_dst='star',
        model_path='data/body_models/star'),
    convention='star',
    loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_vertex=dict(type='L1Loss', loss_weight=2),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=3),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.02))
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx', 'has_smpl',
    'has_keypoints2d', 'has_keypoints3d'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='star'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='RandomErasing'),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

inference_pipeline = [
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation', 'origin_img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='pw3d',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='star',
                ann_file='star.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='star',
                ann_file='mpi_inf_3dhp_1_4.npz'),
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='star',
                ann_file='h36m_train_new.npz'),
        ],
        partition=[0.4, 0.3, 0.3],
    ),
    val=dict(
        type=dataset_type,
        dataset_name='h36m',
        body_model=dict(
            type='STAR',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/star'),
        data_prefix='data',
        pipeline=test_pipeline,
        convention='star',
        ann_file='h36m_test.npz'),
    test=dict(
        type=dataset_type,
        dataset_name='h36m',
        body_model=dict(
            type='STAR',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/star'),
        data_prefix='data',
        pipeline=test_pipeline,
        convention='star',
        ann_file='h36m_test.npz'),
)
