# model settings
model = dict(
    type='HMR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(smpl_mean_params="data/body_models/smpl_mean_params.npz"),
    smpl_pose_criterion=dict(type='MSELoss', loss_weight=1.),
    smpl_betas_criterion=dict(type='MSELoss', loss_weight=0.001))
