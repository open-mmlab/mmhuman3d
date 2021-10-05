_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=2.5e-4),
    head=dict(type='Adam', lr=2.5e-4),
    disc=dict(type='Adam', lr=1e-4))
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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(
        type='HMRHead',
        feat_dim=2048,
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model=dict(
        type='SMPL',
        smpl_path='data/body_models/smpl',
        joints_regressor='data/body_models/joints_regressor_cmr.npy'),
    loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=1000),
    loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=100),
    loss_vertex=dict(type='L1Loss', loss_weight=20),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=30),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.2),
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
    'smpl_transl', 'keypoints2d', 'keypoints2d_mask', 'keypoints3d',
    'keypoints3d_mask'
]
keypoints_index = [_ for _ in range(24)]
flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [20, 21],
              [22, 23]]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='KeypointsSelection', keypoints_index=keypoints_index),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, flip_pairs=flip_pairs),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=224),
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
    dict(type='KeypointsSelection', keypoints_index=keypoints_index),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
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
                    ann_file='h36m_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='mpi_inf_3dhp',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    ann_file='mpi_inf_3dhp_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='lsp_original',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    ann_file='lsp_dataset_original_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='lspet',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    ann_file='hr-lspet_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='mpii',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    ann_file='mpii_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='coco',
                    data_prefix='data',
                    pipeline=train_pipeline,
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
            type='SMPL',
            smpl_path='data/body_models/smpl',
            joints_regressor='data/body_models/joints_regressor_cmr.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
)
