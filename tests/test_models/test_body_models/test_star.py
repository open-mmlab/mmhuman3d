import os

import numpy as np
import torch

from mmhuman3d.models.body_models.builder import build_body_model

BODY_MODEL_LOAD_DIR = 'data/body_models/star'
BETAS = [
    2.25176191, -3.7883464, 0.46747496, 3.89178988, 2.20098416, 0.26102114,
    -3.07428093, 0.55708514, -3.94442258, -2.88552087
]
POSE = [0.1] * 72
TRANSL = [0.2, 0., 0.1]
ORIENT = [-0.1, 0.1, 0.]


def test_star_init():
    _ = build_body_model(dict(type='STAR', model_path=BODY_MODEL_LOAD_DIR))


def test_star_invalid_gender():
    try:
        _ = build_body_model(
            dict(
                type='STAR',
                model_path=BODY_MODEL_LOAD_DIR,
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
                model_path=os.path.join(BODY_MODEL_LOAD_DIR,
                                        'STAR_NEUTRAL.npz'),
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
            model_path=os.path.join(BODY_MODEL_LOAD_DIR, 'STAR_NEUTRAL.npz'),
            gender='neutral'))


def test_star_invalid_path():
    try:
        _ = build_body_model(
            dict(
                type='STAR',
                model_path=os.path.join(BODY_MODEL_LOAD_DIR, 'invalid_path')))
    except RuntimeError as err:
        # NOTE (kristijanbartol): This is one way to check the error type.
        assert ('does not exist' in str(err))


def test_star_specify_other_keypoint_mappings():
    star = build_body_model(
        dict(
            type='STAR',
            model_path=BODY_MODEL_LOAD_DIR,
            keypoint_src='star',
            keypoint_dst='smpl_24'))

    _ = star.forward()


def test_star_create_default_parameters():
    star = build_body_model(
        dict(
            type='STAR',
            model_path=BODY_MODEL_LOAD_DIR,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_transl=True))

    _ = star.forward()


def test_star_init_parameters():
    batch_size = 1
    global_orient = torch.tensor(
        ORIENT, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    body_pose = torch.tensor(
        POSE, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    betas = torch.tensor(
        BETAS, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    transl = torch.tensor(
        TRANSL, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)

    star = build_body_model(
        dict(
            type='STAR',
            model_path=BODY_MODEL_LOAD_DIR,
            create_global_orient=True,
            global_orient=global_orient,
            create_body_pose=True,
            body_pose=body_pose,
            create_betas=True,
            betas=betas,
            create_transl=True,
            transl=transl))

    _ = star.forward()


def test_star_init_parameters_numpy():
    batch_size = 1
    global_orient = np.repeat(
        np.expand_dims(np.array(ORIENT, dtype=np.float32), 0),
        batch_size,
        axis=0)
    body_pose = np.repeat(
        np.expand_dims(np.array(POSE, dtype=np.float32), 0),
        batch_size,
        axis=0)
    betas = np.repeat(
        np.expand_dims(np.array(BETAS, dtype=np.float32), 0),
        batch_size,
        axis=0)
    transl = np.repeat(
        np.expand_dims(np.array(TRANSL, dtype=np.float32), 0),
        batch_size,
        axis=0)

    star = build_body_model(
        dict(
            type='STAR',
            model_path=BODY_MODEL_LOAD_DIR,
            create_global_orient=True,
            global_orient=global_orient,
            create_body_pose=True,
            body_pose=body_pose,
            create_betas=True,
            betas=betas,
            create_transl=True,
            transl=transl))

    _ = star.forward()


def test_star_forward_parameters():
    star = build_body_model(
        dict(
            type='STAR',
            model_path=BODY_MODEL_LOAD_DIR,
        ))

    batch_size = 1
    global_orient = torch.tensor(
        ORIENT, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    body_pose = torch.tensor(
        POSE, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    betas = torch.tensor(
        BETAS, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    transl = torch.tensor(
        TRANSL, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)

    _ = star.forward(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl)


def test_star_init_and_forward_parameters():
    batch_size = 1
    global_orient = torch.zeros(
        (len(ORIENT), ), dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    body_pose = torch.zeros(
        (len(POSE), ), dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    betas = torch.zeros((len(BETAS), ),
                        dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    transl = torch.zeros((len(TRANSL), ),
                         dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)

    star = build_body_model(
        dict(
            type='STAR',
            model_path=BODY_MODEL_LOAD_DIR,
            create_global_orient=True,
            global_orient=global_orient,
            create_body_pose=True,
            body_pose=body_pose,
            create_betas=True,
            betas=betas,
            create_transl=True,
            transl=transl))

    global_orient = torch.tensor(
        ORIENT, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    body_pose = torch.tensor(
        POSE, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    betas = torch.tensor(
        BETAS, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    transl = torch.tensor(
        TRANSL, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)

    _ = star.forward(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl)
