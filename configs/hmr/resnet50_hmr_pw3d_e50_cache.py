_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'])
# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=2.5e-4),
    head=dict(type='Adam', lr=2.5e-4),
    disc=dict(type='Adam', lr=1e-4))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)

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
    head=dict(
        type='HMRHead',
        feat_dim=2048,
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model_train=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_54',
        model_path='data/body_models/smpl',
        keypoint_approximate=True,
        extra_joints_regressor='data/body_models/J_regressor_extra.npy'),
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    convention='smpl_54',
    loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_vertex=dict(type='L1Loss', loss_weight=2),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=3),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.02),
    loss_adv=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1),
    disc=dict(type='SMPLDiscriminator'))
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_54'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
adv_data_keys = [
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
]
train_adv_pipeline = [dict(type='Collect', keys=adv_data_keys, meta_keys=[])]
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
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

cache_files = {
    'h36m': 'data/cache/h36m_mosh_train_smpl_54.npz',
    'mpi_inf_3dhp': 'data/cache/mpi_inf_3dhp_train_smpl_54.npz',
    'lsp': 'data/cache/lsp_train_smpl_54.npz',
    'lspet': 'data/cache/lspet_train_smpl_54.npz',
    'mpii': 'data/cache/mpii_train_smpl_54.npz',
    'coco': 'data/cache/coco_2014_train_smpl_54.npz'
}
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='AdversarialDataset',
        train_dataset=dict(
            type='MixedDataset',
            configs=[
                dict(
                    type=dataset_type,
                    dataset_name='h36m',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    convention='smpl_54',
                    cache_data_path=cache_files['h36m'],
                    ann_file='h36m_mosh_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='mpi_inf_3dhp',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    convention='smpl_54',
                    cache_data_path=cache_files['mpi_inf_3dhp'],
                    ann_file='mpi_inf_3dhp_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='lsp',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    convention='smpl_54',
                    cache_data_path=cache_files['lsp'],
                    ann_file='lsp_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='lspet',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    convention='smpl_54',
                    cache_data_path=cache_files['lspet'],
                    ann_file='lspet_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='mpii',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    convention='smpl_54',
                    cache_data_path=cache_files['mpii'],
                    ann_file='mpii_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='coco',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    convention='smpl_54',
                    cache_data_path=cache_files['coco'],
                    ann_file='coco_2014_train.npz'),
            ],
            partition=[0.35, 0.15, 0.1, 0.10, 0.10, 0.2],
        ),
        adv_dataset=dict(
            type='MeshDataset',
            dataset_name='cmu_mosh',
            data_prefix='data',
            pipeline=train_adv_pipeline,
            ann_file='cmu_mosh.npz')),
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
        ann_file='pw3d_test.npz'),
)
