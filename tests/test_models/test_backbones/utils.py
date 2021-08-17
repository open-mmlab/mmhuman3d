from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmhuman3d.models.backbones.resnet import BasicBlock, Bottleneck
from mmhuman3d.models.utils import SimplifiedBasicBlock


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck, SimplifiedBasicBlock)):
        return True
    return False


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True
