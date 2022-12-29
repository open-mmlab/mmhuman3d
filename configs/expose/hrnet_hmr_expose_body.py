_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

img_res = 256

# evaluate
evaluation = dict(interval=10, metric=['pa-mpjpe', 'mpjpe'])

optimizer = dict(
    backbone=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    head=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[60, 100], gamma=0.1)

runner = dict(type='EpochBasedRunner', max_epochs=100)

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(interval=10)

# model settings
hrnet_extra = dict(
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block='BOTTLENECK',
        num_blocks=(4, ),
        num_channels=(64, )),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(48, 96)),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(48, 96, 192)),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(48, 96, 192, 384)),
    downsample=True,
    use_conv=True,
    pretrained_layers=[
        'conv1',
        'bn1',
        'conv2',
        'bn2',
        'layer1',
        'transition1',
        'stage2',
        'transition2',
        'stage3',
        'transition3',
        'stage4',
    ],
    final_conv_kernel=1,
    return_list=False)

find_unused_parameters = True

model = dict(
    type='SMPLXImageBodyModelEstimator',
    backbone=dict(
        type='PoseHighResolutionNetExpose',
        extra=hrnet_extra,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/pretrained_models/hrnet_pretrain.pth')),
    head=dict(
        type='ExPoseBodyHead',
        num_betas=10,
        num_expression_coeffs=10,
        mean_pose_path='data/body_models/smplx/all_means.pkl',
        shape_mean_path='data/body_models/smplx/shape_mean.npy',
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
        camera_cfg=dict(pos_func='softplus', mean_scale=0.9)),
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
    body_model_test=dict(
        type='SMPLXLayer',
        num_expression_coeffs=10,
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=True,
        model_path='data/body_models/smplx',
        keypoint_src='lsp',
        keypoint_dst='lsp',
        joints_regressor='data/body_models/smplx/SMPLX_to_J14.npy'),
    loss_keypoints3d=dict(type='L1Loss', reduction='sum', loss_weight=1),
    loss_keypoints2d=dict(type='L1Loss', reduction='sum', loss_weight=1),
    loss_smplx_global_orient=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_body_pose=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_jaw_pose=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_hand_pose=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_betas=dict(type='MSELoss', reduction='sum', loss_weight=0.001),
    loss_smplx_expression=dict(type='MSELoss', reduction='sum', loss_weight=1),
    loss_smplx_betas_prior=dict(
        type='ShapeThresholdPriorLoss', margin=3.0, norm='l2', loss_weight=1),
    convention='smplx')

# dataset settings
dataset_type = 'HumanImageSMPLXDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
    'has_smplx_global_orient', 'has_smplx_body_pose', 'has_smplx_jaw_pose',
    'has_smplx_right_hand_pose', 'has_smplx_left_hand_pose', 'has_smplx_betas',
    'has_smplx_expression', 'smplx_jaw_pose', 'smplx_body_pose',
    'smplx_right_hand_pose', 'smplx_left_hand_pose', 'smplx_global_orient',
    'smplx_betas', 'keypoints2d', 'keypoints3d', 'sample_idx',
    'smplx_expression'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BBoxCenterJitter', factor=0.0, dist='normal'),
    dict(type='RandomHorizontalFlip', flip_prob=0.5,
         convention='smplx'),  # hand = 0,head = body = 0.5
    dict(
        type='GetRandomScaleRotation',
        rot_factor=30.0,
        scale_factor=0.25,
        rot_prob=0.6),
    dict(type='MeshAffine', img_res=img_res),  # hand = 224, body = head = 256
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(
        type='SimulateLowRes',
        dist='categorical',
        cat_factors=(1.0, ),
        # head = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 8.0)
        # hand = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0)
        # body = (1.0,)
        factor_min=1.0,
        factor_max=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'ori_img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation', 'ori_img'])
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
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

cache_files = {
    'curated_fits': 'data/cache/curated_fits_train_smplx.npz',
    'spin_smplx': 'data/cache/spin_smplx_train.npz',
    'h36m': 'data/cache/h36m_train_smplx.npz'
}

data = dict(
    samples_per_gpu=48,  # body 48, head = hand = 64
    workers_per_gpu=8,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                pipeline=train_pipeline,
                dataset_name='',
                data_prefix='data',
                ann_file='curated_fits_train.npz',
                convention='smplx',
                num_betas=10,
                num_expression=10,
                cache_data_path=cache_files['curated_fits'],
            ),
            dict(
                type=dataset_type,
                pipeline=train_pipeline,
                dataset_name='',
                data_prefix='data',
                ann_file='spin_smplx_train.npz',
                convention='smplx',
                num_betas=10,
                num_expression=10,
                cache_data_path=cache_files['spin_smplx'],
            ),
            dict(
                type=dataset_type,
                pipeline=train_pipeline,
                dataset_name='h36m',
                data_prefix='data',
                ann_file='h36m_train.npz',
                convention='smplx',
                num_betas=10,
                num_expression=10,
                cache_data_path=cache_files['h36m'],
            ),
        ],
        partition=[0.08, 0.12, 0.8],
    ),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='SMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='3DPW',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='SMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='3DPW',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
)
