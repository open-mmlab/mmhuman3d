from mmhuman3d.data.datasets import build_dataset
from os.path import join

dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'
]

train_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=dict(backend='petrel')),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_54'),
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

data_root = '/mnt/lustre/wangyanjun'
DATASET_NPZ_PATH = join(data_root,'data/dataset_extras')

train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='s3://Zoetrope/OpenHuman/human3.6m',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file=join(DATASET_NPZ_PATH, 'h36m_mmhuman_train.npz')),
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='s3://Zoetrope/OpenHuman/COCO/2014/data/',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file=join(DATASET_NPZ_PATH, 'eft_coco_all.npz')),
            dict(
                type=dataset_type,
                dataset_name='lspet',
                data_prefix='s3://Zoetrope/OpenHuman/hr-lspet/',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file=join(DATASET_NPZ_PATH, 'eft_lspet.npz')),
            dict(
                type=dataset_type,
                dataset_name='mpii',
                data_prefix='s3://Zoetrope/OpenHuman/MPII/data/',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file=join(DATASET_NPZ_PATH, 'eft_mpii.npz')),
            dict(
                type=dataset_type,
                dataset_name='mpi-inf-3dhp',
                data_prefix='s3://Zoetrope/OpenHuman/mpi_inf_3dhp',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file=join(DATASET_NPZ_PATH, 'spin_mpi_inf_3dhp_train_mmhuman.npz')),
            

        ],
        partition=[0.5,0.1,0.1,0.1,0.2],
    )
 



dataset = build_dataset([train])
print(dataset[0])