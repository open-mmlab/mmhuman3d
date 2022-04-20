from mmhuman3d.models.architectures.builder import build_architecture

arch_cfg = dict(
    type='PareImportTestor',
    backbone=dict(
        type='ResNet',
        depth=34,
        out_indices=[3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    head=dict(
        type='HybrIKHead',
        smpl_mean_params='data/body_models/h36m_mean_beta.npy'),
    body_model=dict(
        type='HybrIKSMPL',
        model_path=  # noqa: E251
        'data/body_models/smpl',
        extra_joints_regressor='data/body_models/J_regressor_h36m.npy'),
    loss_beta=dict(type='MSELoss', loss_weight=1),
    loss_theta=dict(type='MSELoss', loss_weight=0.01),
    loss_twist=dict(type='MSELoss', loss_weight=0.2),
    loss_uvd=dict(type='L1Loss', loss_weight=1),
)
arch = build_architecture(arch_cfg)
