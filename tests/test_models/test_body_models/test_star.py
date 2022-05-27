import os

import numpy as np
import torch

from mmhuman3d.models.body_models.builder import build_body_model

body_model_load_dir = 'data/body_models/star'


def test_star_init():
    _ = build_body_model(dict(type='STAR', model_path=body_model_load_dir))


def test_star_invalid_gender():
    try:
        _ = build_body_model(
            dict(
                type='STAR',
                model_path=body_model_load_dir,
                gender='invalid_gender'))
    except RuntimeError as err:
        # NOTE (kristijanbartol): This is one way to check the error type.
        assert ('gender' in str(err))
    except Exception:
        assert (False)


def test_star_incompatible_gender():
    try:
        _ = build_body_model(
            dict(
                type='STAR',
                model_path=os.path.join(body_model_load_dir, 'STAR_MALE.npz'),
                gender='female'))
    except RuntimeError as err:
        # NOTE (kristijanbartol): This is one way to check the error type.
        assert ('incompatible' in str(err))
    except Exception:
        assert (False)


def test_star_correct_full_path():
    _ = build_body_model(
        dict(
            type='STAR',
            model_path=os.path.join(body_model_load_dir, 'STAR_NEUTRAL.npz'),
            gender='neutral'))


def test_star_invalid_path():
    try:
        _ = build_body_model(
            dict(
                type='STAR',
                model_path=os.path.join(body_model_load_dir, 'invalid_path')))
    except RuntimeError as err:
        # NOTE (kristijanbartol): This is one way to check the error type.
        assert ('does not exist' in str(err))


def test_star_forward():
    star = build_body_model(dict(type='STAR', model_path=body_model_load_dir))

    betas = np.array([[
        2.25176191, -3.7883464, 0.46747496, 3.89178988, 2.20098416, 0.26102114,
        -3.07428093, 0.55708514, -3.94442258, -2.88552087
    ]])
    batch_size = 1

    poses = torch.zeros((batch_size, 72), dtype=torch.float)
    betas = torch.tensor(betas, dtype=torch.float)

    trans = torch.zeros((batch_size, 3), dtype=torch.float)
    _ = star.forward(poses, betas, trans)
