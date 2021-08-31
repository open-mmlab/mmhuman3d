from functools import partial
from typing import Iterable, Optional

import numpy as np
import smplx
import torch


def smpl_tensor2dict(full_pose: torch.Tensor,
                     betas: Optional[torch.Tensor] = None,
                     transl: Optional[torch.Tensor] = None):
    NUM_BODY_JOINTS = 23
    full_pose = full_pose.view(-1, (NUM_BODY_JOINTS + 1) * 3)
    body_pose = full_pose[:, 3:]
    global_orient = full_pose[:, :3]
    batch_size = full_pose.shape[0]
    betas = betas.view(batch_size, -1) if betas is not None else betas
    transl = transl.view(batch_size, -1) if transl is not None else transl
    return {
        'betas': betas,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'transl': transl,
    }


def smpl_dict2tensor(smpl_dict):
    NUM_BODY_JOINTS = 23
    global_orient = smpl_dict['global_orient'].view(-1, 3)
    body_pose = smpl_dict['body_pose'].view(-1, 3 * NUM_BODY_JOINTS)
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    return full_pose


def smplx_tensor2dict(full_pose: torch.Tensor,
                      betas: Optional[torch.Tensor] = None,
                      transl: Optional[torch.Tensor] = None,
                      expression: Optional[torch.Tensor] = None):
    NUM_BODY_JOINTS = 23 - 2
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    full_pose = full_pose.view(-1, (NUM_JOINTS + 1), 3)
    global_orient = full_pose[:, :1]
    body_pose = full_pose[:, 1:NUM_BODY_JOINTS + 1]
    jaw_pose = full_pose[:, NUM_BODY_JOINTS + 1:NUM_BODY_JOINTS + 2]
    leye_pose = full_pose[:, NUM_BODY_JOINTS + 2:NUM_BODY_JOINTS + 3]
    reye_pose = full_pose[:, NUM_BODY_JOINTS + 3:NUM_BODY_JOINTS + 4]
    left_hand_pose = full_pose[:, NUM_BODY_JOINTS + 4:NUM_BODY_JOINTS + 19]
    right_hand_pose = full_pose[:, NUM_BODY_JOINTS + 19:NUM_BODY_JOINTS + 34]
    batch_size = body_pose.shape[0]
    betas = betas.view(batch_size, -1) if betas is not None else betas
    transl = transl.view(batch_size, -1) if transl is not None else transl
    expression = expression.view(batch_size,
                                 -1) if expression is not None else expression
    return {
        'betas': betas,
        'global_orient': global_orient.view(batch_size, 3),
        'body_pose': body_pose.view(batch_size, NUM_BODY_JOINTS * 3),
        'left_hand_pose': left_hand_pose.view(batch_size, NUM_HAND_JOINTS * 3),
        'right_hand_pose': right_hand_pose.view(batch_size,
                                                NUM_HAND_JOINTS * 3),
        'transl': transl,
        'expression': expression,
        'jaw_pose': jaw_pose.view(batch_size, 3),
        'leye_pose': leye_pose.view(batch_size, 3),
        'reye_pose': reye_pose.view(batch_size, 3),
    }


def smplx_dict2tensor(smplx_dict):
    NUM_BODY_JOINTS = 23 - 2
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    global_orient = smplx_dict['global_orient'].reshape(-1, 1, 3)
    body_pose = smplx_dict['body_pose'].reshape(-1, NUM_BODY_JOINTS, 3)
    batch_size = global_orient.shape[0]
    jaw_pose = smplx_dict.get('jaw_pose', torch.zeros((batch_size, 1, 3)))
    leye_pose = smplx_dict.get('leye_pose', torch.zeros((batch_size, 1, 3)))
    reye_pose = smplx_dict.get('reye_pose', torch.zeros((batch_size, 1, 3)))
    left_hand_pose = smplx_dict.get(
        'left_hand_pose', torch.zeros((batch_size, NUM_HAND_JOINTS, 3)))
    right_hand_pose = smplx_dict.get(
        'right_hand_pose', torch.zeros((batch_size, NUM_HAND_JOINTS, 3)))
    full_pose = torch.cat([
        global_orient, body_pose,
        jaw_pose.reshape(-1, 1, 3),
        leye_pose.reshape(-1, 1, 3),
        reye_pose.reshape(-1, 1, 3),
        left_hand_pose.reshape(-1, 15, 3),
        right_hand_pose.reshape(-1, 15, 3)
    ],
                          dim=1).reshape(-1, (NUM_JOINTS + 1) * 3)
    return full_pose


def get_body_model(model_folder: str,
                   model_type: str = 'smpl',
                   gender: str = 'neutral',
                   batch_size: int = 1) -> partial:
    return partial(
        smplx.create,
        model_folder,
        model_type=model_type,
        gender=gender,
        use_face_contour=True,
        num_betas=10,
        batch_size=batch_size)


def get_mesh_info(body_model,
                  data_type: str = 'numpy',
                  required_keys: Iterable[str] = ['vertices', 'faces'],
                  **kwposes) -> dict:
    """Get information from smpl(x) body model.

    Args:
        body_model ([type]): smpl(x) body model function.
        data_type (str, optional): wanted data type. Defaults to 'numpy'.
        required_keys (list, optional): The required_keys.
                Defaults to ['vertices', 'faces'].

    Returns:
        dict: return a dict according to required_keys.
    """
    assert isinstance(data_type, str)
    data_type = data_type.lower()
    assert data_type in ['numpy', 'tensor']
    left_hand_pose = kwposes.get('left_hand_pose')
    right_hand_pose = kwposes.get('right_hand_pose')
    if left_hand_pose is not None and right_hand_pose is not None:
        hand_pose_dim = left_hand_pose.shape[-1]
        assert hand_pose_dim == right_hand_pose.shape[-1]
        if hand_pose_dim == 15 * 3:
            body_model = body_model(use_pca=False)
        else:
            body_model = body_model(use_pca=True, num_pca_comps=hand_pose_dim)
    else:
        body_model = body_model()
    model_output = body_model(
        return_verts=True,
        **kwposes,
    )

    res_dict = {}

    if 'vertices' in required_keys:
        if data_type == 'numpy':
            res_dict['vertices'] = model_output.vertices.detach().cpu().numpy()
        else:
            res_dict['vertices'] = model_output.vertices
    if 'faces' in required_keys:
        res_dict['faces'] = body_model.faces
    if 'joints' in required_keys:
        if data_type == 'numpy':
            res_dict['joints'] = model_output.joints.detach().cpu().numpy()
        else:
            res_dict['joints'] = model_output.joints
    if 'parents' in required_keys:
        if data_type == 'numpy':
            res_dict['parents'] = body_model.parents.detach().cpu().numpy()
        else:
            res_dict['parents'] = body_model.parents
    if 'limbs' in required_keys:
        if data_type == 'numpy':
            parents = body_model.parents.detach().cpu().numpy()
            res_dict['limbs'] = np.concatenate(
                [np.arange(parents.shape[0])[np.newaxis], parents[np.newaxis]],
                axis=0).T
        else:
            parents = body_model.parents
            res_dict['limbs'] = torch.cat(
                [torch.arange(parents.shape[0])[None], parents[None]],
                axis=0).T
    return res_dict
