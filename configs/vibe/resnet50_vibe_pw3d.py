_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# optimizer
optimizer = dict(
    neck=dict(type='Adam', lr=5e-5),
    head=dict(type='Adam', lr=5e-5),
    disc=dict(type='Adam', lr=1e-4, weight_decay=1e-4))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=30)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

img_res = 224

# model settings
model = dict(
    type='VideoBodyModelEstimator',
    neck=dict(
        type='TemporalGRUEncoder',
        num_layers=2,
        hidden_size=1024,
        add_linear=True,
        bidirectional=False,
        use_residual=True),
    head=dict(
        type='HMRHead',
        feat_dim=2048,
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model=dict(
        type='SMPL',
        smpl_path='data/body_models/smpl',
        joints_regressor='data/body_models/joints_regressor_cmr.npy',
        extra_joints_regressor='data/body_models/joints_regressor_extra.npy'),
    loss_keypoints3d=dict(type='MSELoss', loss_weight=300),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=300),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=60),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.06),
    # pretrained_head='data/pretrained/new_spin.pt',
    loss_adv=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=0.5),
    disc=dict(
        type='MotionDiscriminator',
        rnn_size=1024,
        input_size=69,
        num_layers=2,
        output_size=1,
        feature_pool='attention',
        attention_size=1024,
        attention_layers=3,
        attention_dropout=0.2))
# dataset settings
dataset_type = 'HumanVideoDataset'
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints2d_mask', 'keypoints3d',
    'keypoints3d_mask', 'features'
]
keypoints_index = [_ for _ in range(49)]
train_pipeline = [
    dict(type='KeypointsSelection', keypoints_index=keypoints_index),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=[*data_keys])
]
adv_data_keys = [
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
]
train_adv_pipeline = [dict(type='Collect', keys=adv_data_keys, meta_keys=[])]

test_meta_keys = ['image_path', 'frame_idx']
test_pipeline = [
    dict(type='Collect', keys=[*data_keys], meta_keys=test_meta_keys)
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
                    dataset_name='mpi_inf_3dhp',
                    data_prefix='data',
                    seq_len=16,
                    pipeline=train_pipeline,
                    ann_file='vibe_mpi_inf_3dhp_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='insta_variety',
                    data_prefix='data',
                    seq_len=16,
                    pipeline=train_pipeline,
                    ann_file='insta_variety.npz'),
            ],
            partition=[0.4, 0.6],
            num_data=16000,
        ),
        adv_dataset=dict(
            type=dataset_type,
            dataset_name='amass',
            data_prefix='data',
            seq_len=16,
            only_vid_name=True,
            pipeline=train_adv_pipeline,
            ann_file='amass.npz'),
    ),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='SMPL',
            smpl_path='data/body_models/smpl',
            joints_regressor='data/body_models/joints_regressor_cmr.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        seq_len=16,
        pipeline=test_pipeline,
        ann_file='vibe_pw3d_test.npz'))
