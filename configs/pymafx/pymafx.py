mmhuman3d_data_path = 'data'
mmhuman3d_config_path = 'configs'

_base_ = [f'../../{mmhuman3d_config_path}/_base_/default_runtime.py']

maf_on = True
__bhf_mode__ = 'full_body'  # full_body or body_hand
__grid_align__ = dict(
    use_att=True,
    use_fc=False,
    att_feat_idx=2,
    att_head=1,
    att_starts=1,
)
__img_res__ = 224
# model settings
__hrnet_extra__ = dict(
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
        global_mode=False), )

__mesh_model__ = dict(
    name='smplx',
    smpl_mean_params=f'{mmhuman3d_data_path}/body_models/smpl_mean_params.npz',
    gender='neutral')

model = dict(
    type='PyMAFX',
    backbone=dict(
        type='PoseHighResolutionNetPyMAFX',
        extra=__hrnet_extra__,
        global_mode=not maf_on,
    ),
    head=dict(
        type='PyMAFXHead',
        maf_on=maf_on,
        n_iter=3,
        bhf_mode=__bhf_mode__,
        grid_feat=False,
        grid_align=__grid_align__,
        mmhuman3d_data_path=mmhuman3d_data_path,
    ),
    regressor=dict(
        type='Regressor',
        mesh_model=__mesh_model__,
        bhf_mode=__bhf_mode__,
        use_iwp_cam=True,
        n_iter=3,
        smpl_model_dir=f'{mmhuman3d_data_path}/body_models/smpl',
        smpl_mean_params=__mesh_model__['smpl_mean_params'],
    ),
    mmhuman3d_data_path=mmhuman3d_data_path,
    attention_config=f'{mmhuman3d_config_path}/pymafx/bert_base_uncased_config.py',
    extra_joints_regressor=f'{mmhuman3d_data_path}/body_models/J_regressor_extra.npy',
    smplx_to_smpl=f'{mmhuman3d_data_path}/body_models/smplx/smplx_to_smpl.npz',
    smplx_model_dir=f'{mmhuman3d_data_path}/body_models/smplx',
    partial_mesh_path=f'{mmhuman3d_data_path}/partial_mesh',
    mesh_model=__mesh_model__,
    bhf_mode=__bhf_mode__,
    maf_on=maf_on,
    body_sfeat_dim=[192, 96, 48],
    hf_sfeat_dim=(256, 256, 256),
    grid_feat=False,
    aux_supv_on=maf_on,
    grid_align=__grid_align__,
    mlp_dim=[256, 128, 64, 5],
    hf_mlp_dim=[256, 128, 64, 5],
    loss_uv_regression_weight=0.5,
    hf_model_cfg=__hf_model_cfg__,
    device='cuda')

# dataset settings
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=False)
dataset_type = 'PyMAFXHumanImageDataset'
inference_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
]
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=8,
    test=dict(
        type=dataset_type,
        data_prefix='data',
        pipeline=inference_pipeline,
        img_res=__img_res__,
        hf_img_size=224),
)
