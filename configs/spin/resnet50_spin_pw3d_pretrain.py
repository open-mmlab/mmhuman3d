_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=3e-5), head=dict(type='Adam', lr=3e-5))

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)
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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(
        type='HMRHead',
        feat_dim=2048,
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model=dict(
        type='SMPL49',
        model_path='data/body_models/smpl',
        gender='neutral',
        create_transl=False,
        extra_joints_regressor='data/J_regressor_extra.npy'),
    # registrant=dict(),
    # loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=1000),
    # loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=100),
    # loss_vertex=dict(type='L1Loss', loss_weight=20),
    # loss_smpl_pose=dict(type='MSELoss', loss_weight=30),
    # loss_smpl_betas=dict(type='MSELoss', loss_weight=0.2),
    loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=5. * 60),
    loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=5. * 60),
    # loss_vertex=dict(type='L1Loss', loss_weight=20),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=1.0 * 60),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.001 * 60),
    # loss_adv=dict(
    #     type='GANLoss',
    #     gan_type='lsgan',
    #     real_label_val=1.0,
    #     fake_label_val=0.0,
    #     loss_weight=1),
    # disc=dict(type='SMPLDiscriminator')
)
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints2d_mask', 'keypoints3d',
    'keypoints3d_mask', 'is_flipped', 'center', 'scale', 'rotation',
    'sample_idx'
]
meta_data_keys = ['dataset_name', 'image_path']
keypoints_index = [_ for _ in range(49)]
# flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [20, 21],
#               [22, 23]]
flip_pairs = [[2, 5], [3, 6], [4, 7], [9, 12], [10, 13], [11, 14], [15, 16],
              [17, 18], [19, 22], [20, 23], [21, 24], [25, 30], [26, 29],
              [27, 28], [31, 36], [32, 35], [33, 34], [45, 46], [47, 48]]
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
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys)
]
# adv_data_keys = [
#     'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
# ]
# train_adv_pipeline = [dict(type='Collect', keys=adv_data_keys, meta_keys=[])]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='KeypointsSelection', keypoints_index=keypoints_index),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys)
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=1,
    train=dict(
        # type='AdversarialDataset',
        # train_dataset=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='data',
                pipeline=train_pipeline,
                ann_file='spin_h36m_train.npz'),
            # dict(
            #     type=dataset_type,
            #     dataset_name='mpi_inf_3dhp',
            #     data_prefix='data',
            #     pipeline=train_pipeline,
            #     ann_file='spin_mpi_inf_3dhp.npz'),
            # dict(
            #     type=dataset_type,
            #     dataset_name='lsp_original',
            #     data_prefix='data',
            #     pipeline=train_pipeline,
            #     ann_file='spin_lsp_dataset_original_train.npz'),
            # dict(
            #     type=dataset_type,
            #     dataset_name='lspet',
            #     data_prefix='data',
            #     pipeline=train_pipeline,
            #     ann_file='spin_hr-lspet_train.npz'),
            # dict(
            #     type=dataset_type,
            #     dataset_name='mpii',
            #     data_prefix='data',
            #     pipeline=train_pipeline,
            #     ann_file='spin_mpii_train.npz'),
            # dict(
            #     type=dataset_type,
            #     dataset_name='coco',
            #     data_prefix='data',
            #     pipeline=train_pipeline,
            #     ann_file='spin_coco_2014_train.npz'),
        ],
        partition=[1.0],
        # ),
        # adv_dataset=dict(
        #     type='MeshDataset',
        #     dataset_name='cmu_mosh',
        #     data_prefix='data',
        #     pipeline=train_adv_pipeline,
        #     ann_file='cmu_mosh.npz')
        # ),
    ),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='SMPL49',
            model_path='data/body_models/smpl',
            gender='neutral',
            create_transl=False,
            extra_joints_regressor='data/J_regressor_extra.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
)
