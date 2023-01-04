_base_ = ['../_base_/default_runtime.py']
__maf_on__ = True
# model settings
__hrnet_extra__ = dict(
    pretr_set='coco',  # 'none' 'imagenet' 'coco'
    pretrained_im='data/pretrained_model/hrnet_w48-imgnet-8ef0771d.pth',
    pretrained_coco='data/pretrained_model/pose_hrnet_w48_256x192.pth',
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(48, 96),
        fuse_method='SUM',
    ),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(48, 96, 192),
        fuse_method='SUM'),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(48, 96, 192, 384),
        fuse_method='SUM'),
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
)

__hf_model_cfg__ = dict(
    backbone=dict(
        type='PoseResNet',
        extra=dict(
            deconv_with_bias=False,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
            num_layers=50),
        global_mode=not __maf_on__), )

model = dict(
    type='PyMAFX',
    backbone=dict(
        type='PoseHighResolutionNetPyMAFX',
        extra=__hrnet_extra__,
        global_mode=not __maf_on__,
    ),
    head=dict(
        type='PyMAFXHead',
        maf_on=__maf_on__,
        n_iter=3,
        bhf_mode='full_body',
        grid_align=dict(
            use_att=True,
            use_fc=False,
            att_feat_idx=2,
            att_head=1,
            att_starts=1,
        ),
    ),
    attention_config='configs/pymafx/bert_base_uncased_config.json',
    joint_regressor_train_extra='data/body_models/J_regressor_extra.npy',
    smpl_model_dir='data/body_models/smpl',
    mesh_model=dict(
        name='smplx',
        smpl_mean_params='data/body_models/smpl_mean_params.npz',
        gender='neutral'),
    bhf_mode='full_body',  # full_body or body_hand
    maf_on=__maf_on__,
    body_sfeat_dim=[192, 96, 48],
    hf_sfeat_dim=(256, 256, 256),
    grid_feat=False,
    grid_align=dict(
        use_att=True,
        use_fc=False,
        att_feat_idx=2,
        att_head=1,
        att_starts=1,
    ),
    mlp_dim=[256, 128, 64, 5],
    hf_mlp_dim=[256, 128, 64, 5],
    loss_uv_regression_weight=0.5,
    hf_model_cfg=__hf_model_cfg__)

# dataset settings
dataset_type = 'PyMAFXHumanImageDataset'
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=8,
    test=dict(type=dataset_type, ),
)
