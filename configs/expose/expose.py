_base_ = ['../_base_/default_runtime.py']
__face_model_checkpoint__ = 'data/pretrained_models/'
'resnet18_hmr_expose_face.pth'
__hand_model_checkpoint__ = 'data/pretrained_models/'
'resnet18_hmr_expose_hand.pth'
__body_model_checkpoint__ = 'data/pretrained_models/hrnet_hmr_expose_body.pth'
__mean_pose_path__ = 'data/body_models/all_means.pkl'
__model_path__ = 'data/body_models/smplx'
__joints_regressor__ = 'data/body_models/smplx/SMPLX_to_J14.npy'

use_adversarial_train = True
img_res = 256

# evaluate
evaluation = dict(
    interval=10,
    metric=['pa-mpjpe', 'pa-pve'],
    body_part=[['body', 'right_hand', 'left_hand'],
               ['', 'right_hand', 'left_hand', 'face']])

optimizer = dict(
    backbone=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    head=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    hand_backbone=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    face_backbone=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    hand_head=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    face_head=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[60, 100], gamma=0.1)

runner = dict(type='EpochBasedRunner', max_epochs=100)

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(interval=10)
face_vertex_ids_path = 'data/body_models/smplx/SMPL-X__FLAME_vertex_ids.npy'
hand_vertex_ids_path = 'data/body_models/smplx/MANO_SMPLX_vertex_ids.pkl'
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

extra_hand_model_cfg = dict(
    backbone=dict(
        type='ResNet',
        depth=18,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=__hand_model_checkpoint__,
            prefix='backbone')),
    head=dict(
        type='ExPoseHandHead',
        num_betas=10,
        mean_pose_path=__mean_pose_path__,
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint=__hand_model_checkpoint__,
            prefix='head')),
    crop_cfg=dict(
        img_res=img_res,
        scale_factor=3.0,
        crop_size=224,
        condition_hand_wrist_pose=True,
        condition_hand_shape=False,
        condition_hand_finger_pose=True,
    ),
    loss_hand_crop=dict(type='L1Loss', reduction='sum', loss_weight=1),
)

extra_face_model_cfg = dict(
    backbone=dict(
        type='ResNet',
        depth=18,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=__face_model_checkpoint__,
            prefix='backbone')),
    head=dict(
        type='ExPoseFaceHead',
        num_betas=100,
        num_expression_coeffs=50,
        mean_pose_path=__mean_pose_path__,
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint=__face_model_checkpoint__,
            prefix='head'),
    ),
    crop_cfg=dict(
        img_res=img_res,
        scale_factor=2.0,
        crop_size=256,
        num_betas=10,
        num_expression_coeffs=10,
        condition_face_neck_pose=False,
        condition_face_jaw_pose=True,
        condition_face_shape=False,
        condition_face_expression=True),
    loss_face_crop=dict(type='L1Loss', reduction='sum', loss_weight=1),
)

find_unused_parameters = True

model = dict(
    type='SMPLXImageBodyModelEstimator',
    backbone=dict(
        type='PoseHighResolutionNetExpose',
        extra=hrnet_extra,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=__body_model_checkpoint__,
            prefix='backbone')),
    head=dict(
        type='ExPoseBodyHead',
        num_betas=10,
        num_expression_coeffs=10,
        mean_pose_path=__mean_pose_path__,
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
        camera_cfg=dict(pos_func='softplus', mean_scale=0.9),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=__body_model_checkpoint__,
            prefix='head'),
    ),
    body_model_train=dict(
        type='SMPLXLayer',
        num_expression_coeffs=10,
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=True,
        model_path=__model_path__,
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
        model_path=__model_path__,
        keypoint_src='lsp',
        keypoint_dst='lsp',
        joints_regressor=__joints_regressor__),
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
    extra_hand_model_cfg=extra_hand_model_cfg,
    extra_face_model_cfg=extra_face_model_cfg,
    frozen_batchnorm=True,
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
    dict(type='Rotation'),
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
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'ori_img',
            'crop_transform'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'ori_img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'ori_img',
            'crop_transform'
        ])
]
inference_pipeline = [
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'ori_img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'ori_img',
            'crop_transform'
        ])
]

cache_files = {
    'curated_fits': 'data/cache/curated_fits_train_smplx.npz',
}

data = dict(
    samples_per_gpu=48,
    workers_per_gpu=8,
    train=dict(
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
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            model_path=__model_path__,
            joints_regressor=__joints_regressor__),
        dataset_name='EHF',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='ehf_val.npz',
        face_vertex_ids_path=face_vertex_ids_path,
        hand_vertex_ids_path=hand_vertex_ids_path,
        convention='smplx'),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            model_path=__model_path__,
            joints_regressor=__joints_regressor__),
        dataset_name='EHF',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='ehf_val.npz',
        face_vertex_ids_path=face_vertex_ids_path,
        hand_vertex_ids_path=hand_vertex_ids_path,
        convention='smplx'),
)
