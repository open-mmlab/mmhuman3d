# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .balanced_mse_loss import BMCLossMD
from .cross_entropy_loss import CrossEntropyLoss
from .gan_loss import GANLoss
from .mse_loss import KeypointMSELoss, MSELoss
from .prior_loss import (
    CameraPriorLoss,
    JointPriorLoss,
    LimbLengthLoss,
    MaxMixturePrior,
    PoseRegLoss,
    ShapePriorLoss,
    ShapeThresholdPriorLoss,
    SmoothJointLoss,
    SmoothPelvisLoss,
    SmoothTranslationLoss,
)
from .rotaion_distance_loss import RotationDistance
from .smooth_l1_loss import L1Loss, SmoothL1Loss

LOSSES = Registry('losses')

LOSSES.register_module(name='GANLoss', module=GANLoss)
LOSSES.register_module(name='MSELoss', module=MSELoss)
LOSSES.register_module(name='KeypointMSELoss', module=KeypointMSELoss)
LOSSES.register_module(name='ShapePriorLoss', module=ShapePriorLoss)
LOSSES.register_module(name='PoseRegLoss', module=PoseRegLoss)
LOSSES.register_module(name='LimbLengthLoss', module=LimbLengthLoss)
LOSSES.register_module(name='JointPriorLoss', module=JointPriorLoss)
LOSSES.register_module(name='SmoothJointLoss', module=SmoothJointLoss)
LOSSES.register_module(name='SmoothPelvisLoss', module=SmoothPelvisLoss)
LOSSES.register_module(
    name='SmoothTranslationLoss', module=SmoothTranslationLoss)
LOSSES.register_module(
    name='ShapeThresholdPriorLoss', module=ShapeThresholdPriorLoss)
LOSSES.register_module(name='CameraPriorLoss', module=CameraPriorLoss)
LOSSES.register_module(name='MaxMixturePrior', module=MaxMixturePrior)
LOSSES.register_module(name='L1Loss', module=L1Loss)
LOSSES.register_module(name='SmoothL1Loss', module=SmoothL1Loss)
LOSSES.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSSES.register_module(name='RotationDistance', module=RotationDistance)
LOSSES.register_module(name='BMCLossMD', module=BMCLossMD)


def build_loss(cfg):
    """Build loss."""
    if cfg is None:
        return None
    return LOSSES.build(cfg)
