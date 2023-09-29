_base_ = ['../_base_/default_runtime.py']
__model_path__ = 'data/body_models/smplx'
__joints_regressor__ = 'data/body_models/smplx/SMPLX_NEUTRAL.npz'


# quick accessA
save_epoch = 1
lr = 1e-5
min_lr = 5e-7
end_epoch = 5
train_batch_size = 16

syncbn = True
bbox_ratio = 1.2

img_res=(384, 512)
input_body_shape = (256, 192)
princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)
model = dict(
    type='SMPLer_X',
    backbone=dict(
        type='ViT',
        img_size=input_body_shape,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        init_cfg=dict(type='Pretrained', checkpoint='data/pretrained_models/vitpose_huge.pth')
    ),
    focal = (5000, 5000),
    camera_3d_size = 2.5,
    input_img_shape =img_res,
    input_body_shape = input_body_shape,
    input_hand_shape = (256, 256),
    input_face_shape = (192, 192),
    output_hm_shape = (16, 16, 12),
    output_hand_hm_shape = (16, 16, 16),
    output_face_hm_shape = (8, 8, 8),
    testset = 'EHF',
    princpt = princpt,
    feat_dim = 1280,
    upscale = 4,
)
# dataset settings
dataset_type = 'HumanImageSMPLXDataset'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
data_keys = []
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
test_pipeline = []
inference_pipeline = [
    dict(type='MeshAffine', img_res=img_res),
    dict(type='ImageToTensor', keys=['img', 'ori_img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idSx'],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'ori_img',
            'crop_transform'
        ])
]


data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(),
    test=dict(),
    val=dict(),
)


# dataset setting
agora_fix_betas = True
agora_fix_global_orient_transl = True
agora_valid_root_pose = True

# for ubody ft
dataset_list = ['Human36M', 'MSCOCO', 'MPII', 'AGORA', 'EHF', 'SynBody', 'GTA_Human2', \
    'EgoBody_Egocentric', 'EgoBody_Kinect', 'UBody', 'PW3D', 'MuCo', 'PROX']
trainset_3d = ['PW3D']
trainset_2d = []
trainset_humandata = ['GTA_Human2', 'EgoBody_Kinect', 'InstaVariety', 'HumanSC3D']


use_cache = True

# strategy
data_strategy = 'balance' # 'balance' need to define total_data_len
total_data_len = 1000000 # assign number or 'auto' for concat length

Talkshow_train_sample_interval = 10

# fine-tune
fine_tune = None # 'backbone', 'head', None for full network tuning

smplx_loss_weight = 1.0 #2 for agora_model for smplx shape
smplx_pose_weight = 10.0

smplx_kps_3d_weight = 100.0
smplx_kps_2d_weight = 1.0
net_kps_2d_weight = 1.0




## =====FIXED ARGS============================================================
## model setting
hand_pos_joint_num = 20
face_pos_joint_num = 72
num_task_token = 24
num_noise_sample = 0

## UBody setting
train_sample_interval = 10
test_sample_interval = 100
make_same_len = False

## input, output size

body_3d_size = 2
hand_3d_size = 0.3
face_3d_size = 0.3
camera_3d_size = 2.5




