"""pytest tests/test_forward.py."""
import copy
from os.path import dirname, exists, join

import torch

from mmhuman3d.models import build_framework


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmhuman3d
        repo_dpath = dirname(dirname(mmhuman3d.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_framework_cfg(fname):
    """Grab configs necessary to create a framework.
    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def test_hmr_forward():
    config_path = 'hmr/resnet50_hmr.py'
    model = _get_framework_cfg(config_path)
    model.backbone.init_cfg = None
    model.head.smpl_mean_params = "tests/data/weights/smpl_mean_params.npz"
    model = build_framework(model)
    model.init_weights()
    input_shape = (1, 3, 224, 224)
    img = torch.rand(input_shape)
    img_metas = {}
    has_smpl = torch.FloatTensor([1])
    smpl_body_pose = torch.randn(1, 23, 3)
    smpl_global_orient = torch.randn(1, 3)
    smpl_betas = torch.rand(1, 10)
    # Test forward train
    model.train()
    losses = model.forward(
        img=img,
        img_metas=img_metas,
        has_smpl=has_smpl,
        smpl_body_pose=smpl_body_pose,
        smpl_global_orient=smpl_global_orient,
        smpl_betas=smpl_betas)
    assert isinstance(losses, dict)
    loss, _ = model._parse_losses(losses)
    assert float(loss.item()) > 0
