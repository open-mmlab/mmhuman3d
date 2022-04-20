
use_adversarial_train = True

# evaluate
evaluation = dict(interval=10, metric=['pa-mpjpe', 'mpjpe'])
# optimizer
# optimizer = dict(type='Adam', lr=5.0e-05)
optimizer = dict(type='Adam', lr=2.0e-04)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=200)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


_base_ = ['../_base_/default_runtime.py']
checkpoint_config = dict(interval=10)
width=32
downsample=False
use_conv=True
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
        num_channels=(width, width*2)),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(width, width*2, width*4)),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(width, width*2, width*4, width*8)),
    downsample = downsample,
    use_conv = use_conv,
    pretrained_layers = [
        'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1',
        'stage2', 'transition2', 'stage3', 'transition3', 'stage4',
    ],
    final_conv_kernel = 1,

    )


find_unused_parameters = True

model = dict(
    type='PARE',
    backbone=dict(
            type='PoseHighResolutionNet',
            extra = hrnet_extra,
            num_joints = 24, 
            init_cfg=dict(type='Pretrained', checkpoint='/mnt/lustre/wangyanjun/data/pretrained_models/pose_coco/pose_hrnet_w32_256x192.pth')
            ),
    head=dict(
        type='PareHead',
        num_joints = 24,
        num_input_features = 480, 
        smpl_mean_params='/mnt/lustre/wangyanjun/data/smpl_mean_params.npz',
        num_deconv_layers = 2,
        num_deconv_filters = [128]*2, # num_deconv_filters = [num_deconv_filters] * num_deconv_layers
        num_deconv_kernels = [4]*2, # num_deconv_kernels = [num_deconv_kernels] * num_deconv_layers
        use_heatmaps = 'part_segm',
        use_keypoint_attention = True,
        backbone = 'hrnet_w32-conv',
        ),
    body_model_train=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_49',
        model_path='/mnt/lustre/wangyanjun/data/body_models/smpl',
        keypoint_approximate=True,

        extra_joints_regressor='/mnt/lustre/wangyanjun/data/J_regressor_extra.npy'),
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='/mnt/lustre/wangyanjun/data/body_models/smpl',
        joints_regressor='/mnt/lustre/wangyanjun/data/J_regressor_h36m.npy'),
    convention='smpl_49',
    loss_convention='smpl_24',
    loss_keypoints3d=dict(type='MSELoss', loss_weight=300),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=300),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=60),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=60*0.001),
    loss_segm_mask=dict(type='CrossEntropyLoss', loss_weight=60),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=1),


)




# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl','has_keypoints3d', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'
]
train_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=dict(backend='petrel')),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='SyntheticOcclusion',pascal_voc_root_path='/mnt/lustre/wangyanjun/data/VOCdevkit/VOC2012/',occluders_file='/mnt/lustre/wangyanjun/data/pascal_occluders.pkl'),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_49'),
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
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

inference_pipeline = [
    dict(type='MeshAffine', img_res=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]


data_root = '/mnt/lustre/wangyanjun/'
DATASET_NPZ_PATH = data_root+'data/dataset_extras/'

cache_files = {
    'h36m': 'data/cache/h36m_mosh_train_smpl_49.npz',
    'mpi-inf-3dhp': 'data/cache/mpi_inf_3dhp_train_smpl_49.npz',
    'lsp': 'data/cache/lsp_train_smpl_49.npz',
    'lspet': 'data/cache/lspet_train_smpl_49.npz',
    'mpii': 'data/cache/mpii_train_smpl_49.npz',
    'coco': 'data/cache/coco_2014_train_smpl_49.npz'
}

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=0,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='s3://Zoetrope/OpenHuman/COCO/2014/data/',
                pipeline=train_pipeline,
                convention='smpl_49',
                cache_data_path=cache_files['coco'],
                ann_file=DATASET_NPZ_PATH+ 'eft_coco_all.npz'),
            

        ],
        partition=[1.0],
    ),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='/mnt/lustre/wangyanjun/data/body_models/smpl',
            joints_regressor='/mnt/lustre/wangyanjun/data/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='/mnt/lustre/wangyanjun/data/dataset_folders/3dpw',
        pipeline=test_pipeline,
        ann_file='/mnt/lustre/wangyanjun/data/dataset_extras_mmhuman/pw3d_test.npz'),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='/mnt/lustre/wangyanjun/data/body_models/smpl',
            joints_regressor='/mnt/lustre/wangyanjun/data/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='/mnt/lustre/wangyanjun/data/dataset_folders/3dpw',
        pipeline=test_pipeline,
        ann_file='/mnt/lustre/wangyanjun/data/dataset_extras_mmhuman/pw3d_test.npz'),
)

