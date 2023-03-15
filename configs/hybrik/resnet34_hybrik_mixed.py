_base_ = ['../_base_/default_runtime.py']

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'])
# optimizer
optimizer = dict(type='Adam', lr=1e-3, weight_decay=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[180, 240])
runner = dict(type='EpochBasedRunner', max_epochs=400)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #    dict(type='TensorboardLoggerHook')
    ])

img_res = 256

# model settings
model = dict(
    type='HybrIK_trainer',
    backbone=dict(
        type='ResNet',
        depth=34,
        out_indices=[3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    head=dict(
        type='HybrIKHead',
        smpl_mean_params='data/body_models/h36m_mean_beta.npy'),
    body_model=dict(
        type='HybrIKSMPL',
        model_path=  # noqa: E251
        'data/body_models/smpl',
        extra_joints_regressor='data/body_models/J_regressor_h36m.npy'),
    loss_beta=dict(type='MSELoss', loss_weight=1),
    loss_theta=dict(type='MSELoss', loss_weight=0.01),
    loss_twist=dict(type='MSELoss', loss_weight=0.2),
    loss_uvd=dict(type='L1Loss', loss_weight=1),
)

# dataset settings
dataset_type = 'HybrIKHumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_keys = [
    'trans_inv', 'intrinsic_param', 'joint_root', 'depth_factor',
    'target_uvd_29', 'target_xyz_24', 'target_weight_24', 'target_weight_29',
    'target_xyz_17', 'target_weight_17', 'target_theta', 'target_beta',
    'target_smpl_weight', 'target_theta_weight', 'target_twist',
    'target_twist_weight', 'bbox', 'sample_idx'
]

h36m_idxs = [
    148, 145, 4, 7, 144, 5, 8, 150, 146, 152, 147, 16, 18, 20, 17, 19, 21
]
hybrik29_idxs = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 16, 17, 18, 19, 20,
    21, 66, 71, 15, 68, 73, 60, 63
]
flip_pairs = [[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17], [18, 19],
              [20, 21], [23, 24], [25, 40], [26, 41], [27, 42], [28, 43],
              [29, 44], [30, 45], [31, 46], [32, 47], [33, 48], [34, 49],
              [35, 50], [36, 51], [37, 52], [38, 53], [39, 54], [57, 56],
              [59, 58], [60, 63], [61, 64], [62, 65], [66, 71], [67, 72],
              [68, 73], [69, 74], [70, 75], [81, 80], [82, 79], [83, 78],
              [84, 77], [85, 76], [101, 98], [102, 97], [103, 96], [104, 95],
              [105, 100], [106, 99], [145, 144], [158, 155], [159, 156],
              [160, 157], [165, 162], [166, 163], [167, 164], [169, 168],
              [171, 170], [172, 175], [173, 176], [174, 177], [179, 180],
              [181, 182], [183, 184], [188, 189]]

keypoints_maps = [
    dict(
        keypoints=[
            'keypoints3d17',
            'keypoints3d17_vis',
            'keypoints3d17_relative',
        ],
        keypoints_index=h36m_idxs),
    dict(
        keypoints=['keypoints3d', 'keypoints3d_vis', 'keypoints3d_relative'],
        keypoints_index=hybrik29_idxs),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomDPG', dpg_prob=0.9),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='RandomOcclusion', occlusion_prob=0.9),
    dict(type='HybrIKRandomFlip', flip_prob=0.5, flip_pairs=flip_pairs),
    dict(type='NewKeypointsSelection', maps=keypoints_maps),
    dict(type='HybrIKAffine', img_res=img_res),
    dict(type='GenerateHybrIKTarget', img_res=img_res, test_mode=False),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['center', 'scale', 'rotation', 'image_path'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NewKeypointsSelection', maps=keypoints_maps),
    dict(type='HybrIKAffine', img_res=img_res),
    dict(type='GenerateHybrIKTarget', img_res=img_res, test_mode=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['center', 'scale', 'rotation', 'image_path'])
]

hp3d_idxs = [
    147, 146, 17, 19, 21, 16, 18, 20, 144, 5, 8, 145, 4, 7, 148, 150, 152
]
hp3d_keypoints_map = [
    dict(
        keypoints=['keypoints3d', 'keypoints3d_vis', 'keypoints3d_relative'],
        keypoints_index=hp3d_idxs),
]

test_hp3d_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NewKeypointsSelection', maps=hp3d_keypoints_map),
    dict(type='HybrIKAffine', img_res=img_res),
    dict(type='GenerateHybrIKTarget', img_res=img_res, test_mode=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['center', 'scale', 'rotation', 'image_path'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='data',
                pipeline=train_pipeline,
                ann_file='hybrik_h36m_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                ann_file='hybrik_mpi_inf_3dhp_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='data',
                pipeline=train_pipeline,
                ann_file='hybrik_coco_2017_train.npz'),
        ],
        partition=[0.4, 0.1, 0.5]),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL', model_path='data/body_models/smpl'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='hybrik_pw3d_test.npz'),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL', model_path='data/body_models/smpl'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='hybrik_pw3d_test.npz'),
)
