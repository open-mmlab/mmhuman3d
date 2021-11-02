import copy
import os.path as osp
from typing import Iterable, Optional, Union

import numpy as np
import smplx
import torch

from mmhuman3d.core.conventions.cameras import (
    convert_cameras,
    convert_perspective_to_weakperspective,
    convert_world_view,
)
from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa


class SMPL_(smplx.SMPL):
    body_pose_keys = {
        'global_orient',
        'body_pose',
    }
    full_pose_keys = {
        'global_orient',
        'body_pose',
    }

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.torch.Tensor,
                    betas: Optional[torch.torch.Tensor] = None,
                    transl: Optional[torch.torch.Tensor] = None):
        full_pose = full_pose.view(-1, (cls.NUM_BODY_JOINTS + 1) * 3)
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

    @classmethod
    def dict2tensor(cls, smpl_dict):
        assert cls.body_pose_keys.issubset(smpl_dict)
        global_orient = smpl_dict['global_orient'].view(-1, 3)
        body_pose = smpl_dict['body_pose'].view(-1, 3 * cls.NUM_BODY_JOINTS)
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        return full_pose


class SMPLX_(smplx.SMPLX):
    body_pose_keys = {'global_orient', 'body_pose'}
    full_pose_keys = {
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose'
    }

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.torch.Tensor,
                    betas: Optional[torch.torch.Tensor] = None,
                    transl: Optional[torch.torch.Tensor] = None,
                    expression: Optional[torch.torch.Tensor] = None):
        NUM_BODY_JOINTS = cls.NUM_BODY_JOINTS
        NUM_HAND_JOINTS = cls.NUM_HAND_JOINTS
        NUM_FACE_JOINTS = cls.NUM_FACE_JOINTS
        NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
        full_pose = full_pose.view(-1, (NUM_JOINTS + 1), 3)
        global_orient = full_pose[:, :1]
        body_pose = full_pose[:, 1:NUM_BODY_JOINTS + 1]
        jaw_pose = full_pose[:, NUM_BODY_JOINTS + 1:NUM_BODY_JOINTS + 2]
        leye_pose = full_pose[:, NUM_BODY_JOINTS + 2:NUM_BODY_JOINTS + 3]
        reye_pose = full_pose[:, NUM_BODY_JOINTS + 3:NUM_BODY_JOINTS + 4]
        left_hand_pose = full_pose[:, NUM_BODY_JOINTS + 4:NUM_BODY_JOINTS + 19]
        right_hand_pose = full_pose[:,
                                    NUM_BODY_JOINTS + 19:NUM_BODY_JOINTS + 34]
        batch_size = body_pose.shape[0]
        betas = betas.view(batch_size, -1) if betas is not None else betas
        transl = transl.view(batch_size, -1) if transl is not None else transl
        expression = expression.view(
            batch_size, -1) if expression is not None else expression
        return {
            'betas':
            betas,
            'global_orient':
            global_orient.view(batch_size, 3),
            'body_pose':
            body_pose.view(batch_size, NUM_BODY_JOINTS * 3),
            'left_hand_pose':
            left_hand_pose.view(batch_size, NUM_HAND_JOINTS * 3),
            'right_hand_pose':
            right_hand_pose.view(batch_size, NUM_HAND_JOINTS * 3),
            'transl':
            transl,
            'expression':
            expression,
            'jaw_pose':
            jaw_pose.view(batch_size, 3),
            'leye_pose':
            leye_pose.view(batch_size, 3),
            'reye_pose':
            reye_pose.view(batch_size, 3),
        }

    @classmethod
    def dict2tensor(cls, smplx_dict):
        assert cls.body_pose_keys.issubset(smplx_dict)
        NUM_BODY_JOINTS = cls.NUM_BODY_JOINTS
        NUM_HAND_JOINTS = cls.NUM_HAND_JOINTS
        NUM_FACE_JOINTS = cls.NUM_FACE_JOINTS
        NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
        global_orient = smplx_dict['global_orient'].reshape(-1, 1, 3)
        body_pose = smplx_dict['body_pose'].reshape(-1, NUM_BODY_JOINTS, 3)
        batch_size = global_orient.shape[0]
        jaw_pose = smplx_dict.get('jaw_pose', torch.zeros((batch_size, 1, 3)))
        leye_pose = smplx_dict.get('leye_pose', torch.zeros(
            (batch_size, 1, 3)))
        reye_pose = smplx_dict.get('reye_pose', torch.zeros(
            (batch_size, 1, 3)))
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

    def forward(self,
                betas: Optional[torch.Tensor] = None,
                global_orient: Optional[torch.Tensor] = None,
                body_pose: Optional[torch.Tensor] = None,
                left_hand_pose: Optional[torch.Tensor] = None,
                right_hand_pose: Optional[torch.Tensor] = None,
                transl: Optional[torch.Tensor] = None,
                expression: Optional[torch.Tensor] = None,
                jaw_pose: Optional[torch.Tensor] = None,
                leye_pose: Optional[torch.Tensor] = None,
                reye_pose: Optional[torch.Tensor] = None,
                return_verts: bool = True,
                return_full_pose: bool = False,
                pose2rot: bool = True,
                return_shaped: bool = True,
                **kwargs):
        global_orient = (
            global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        left_hand_pose = (
            left_hand_pose
            if left_hand_pose is not None else self.left_hand_pose)
        right_hand_pose = (
            right_hand_pose
            if right_hand_pose is not None else self.right_hand_pose)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])
        scale = int(batch_size / self.expression.shape[0])
        expression = expression if expression is not None else self.expression
        expression = self.expression.expand(scale, -1)[:batch_size]

        return super().forward(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=return_verts,
            return_full_pose=return_full_pose,
            pose2rot=pose2rot,
            return_shaped=return_shaped,
            **kwargs)


def create(
        model_path: str,
        model_type: str = 'smpl',
        **kwargs
) -> Union[SMPL_, smplx.SMPLH, SMPLX_, smplx.MANO, smplx.FLAME]:
    """Method for creating a model from a path and a model type.

    Parameters
    ----------
    model_path: str
        Either the path to the model you wish to load or a folder,
        where each subfolder contains the differents types, i.e.:
        model_path:
        |
        |-- smpl
            |-- SMPL_FEMALE
            |-- SMPL_NEUTRAL
            |-- SMPL_MALE
        |-- smplh
            |-- SMPLH_FEMALE
            |-- SMPLH_MALE
        |-- smplx
            |-- SMPLX_FEMALE
            |-- SMPLX_NEUTRAL
            |-- SMPLX_MALE
        |-- mano
            |-- MANO RIGHT
            |-- MANO LEFT

    model_type: str, optional
        When model_path is a folder, then this parameter specifies  the
        type of model to be loaded
    **kwargs: dict
        Keyword arguments

    Returns
    -------
        body_model: nn.Module
            The PyTorch module that implements the corresponding body model
    Raises
    ------
        ValueError: In case the model type is not one of SMPL, SMPLH,
        SMPLX, MANO or FLAMEpartial
    """

    # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = osp.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPL_(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return smplx.SMPLH(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLX_(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return smplx.MANO(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return smplx.FLAME(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')


def get_body_model(model_path: str,
                   model_type: str = 'smpl',
                   gender: str = 'neutral',
                   batch_size: int = 1,
                   num_betas: int = 10,
                   use_face_contour=True,
                   use_pca=False,
                   num_pca_comps=6):
    return create(
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        use_face_contour=use_face_contour,
        num_betas=num_betas,
        batch_size=batch_size,
        use_pca=use_pca,
        num_pca_comps=num_pca_comps)


def get_mesh_info(body_model,
                  use_numpy: bool = False,
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
    model_output = body_model(
        return_verts=True,
        **kwposes,
    )

    res_dict = {}
    allowed_keys = {'vertices', 'faces', 'joints', 'parents', 'limbs'}
    assert set(required_keys).issubset(
        allowed_keys
    ), f'Make sure the required_keys are all in {allowed_keys}. '
    f'Your required_keys: {required_keys}.'
    if 'vertices' in required_keys:
        if use_numpy:
            res_dict['vertices'] = model_output.vertices.detach().cpu().numpy()
        else:
            res_dict['vertices'] = model_output.vertices
    if 'faces' in required_keys:
        if use_numpy:
            res_dict['faces'] = body_model.faces
        else:
            res_dict['faces'] = torch.Tensor(body_model.faces.astype(np.int32))
    if 'joints' in required_keys:
        if use_numpy:
            res_dict['joints'] = model_output.joints.detach().cpu().numpy()
        else:
            res_dict['joints'] = model_output.joints
    if 'parents' in required_keys:
        if use_numpy:
            res_dict['parents'] = body_model.parents.detach().cpu().numpy()
        else:
            res_dict['parents'] = body_model.parents
    if 'limbs' in required_keys:
        if use_numpy:
            parents = body_model.parents.detach().cpu().numpy()
            res_dict['limbs'] = np.concatenate(
                [np.arange(parents.shape[0])[np.newaxis], parents[np.newaxis]],
                axis=0).T
        else:
            parents = body_model.parents
            parents = torch.tensor(parents)
            res_dict['limbs'] = torch.cat(
                [torch.arange(parents.shape[0])[None], parents[None]], dim=0).T
    return res_dict


def convert_smpl_from_opencv_calibration(
        R: Union[np.ndarray, torch.Tensor],
        T: Union[np.ndarray, torch.Tensor],
        K: Optional[Union[np.ndarray, torch.Tensor]] = None,
        resolution: Optional[Union[Iterable[int], int]] = None,
        verts: Optional[Union[np.ndarray, torch.Tensor]] = None,
        poses: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
        body_model_dir: Optional[str] = None,
        betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
        model_type: Optional[str] = 'smpl',
        gender: Optional[str] = 'neutral'):
    """Convert opencv calibration smpl poses&transl parameters to model based
    poses&transl or verts.

    Args:
        R (Union[np.ndarray, torch.Tensor]): (frame, 3, 3)
        T (Union[np.ndarray, torch.Tensor]): [(frame, 3)
        K (Optional[Union[np.ndarray, torch.Tensor]], optional):
            (frame, 3, 3) or (frame, 4, 4). Defaults to None.
        resolution (Optional[Union[Iterable[int], int]], optional):
            (height, width). Defaults to None.
        verts (Optional[Union[np.ndarray, torch.Tensor]], optional):
            (frame, num_verts, 3). Defaults to None.
        poses (Optional[Union[np.ndarray, torch.Tensor]], optional):
            (frame, 72/165). Defaults to None.
        transl (Optional[Union[np.ndarray, torch.Tensor]], optional):
            (frame, 3). Defaults to None.
        body_model_dir (Optional[str], optional): model path.
            Defaults to None.
        betas (Optional[Union[np.ndarray, torch.Tensor]], optional):
            (frame, 10). Defaults to None.
        model_type (Optional[str], optional): choose in 'smpl' or 'smplx'.
            Defaults to 'smpl'.
        gender (Optional[str], optional): choose in 'male', 'female',
            'neutral'.
            Defaults to 'neutral'.

    Raises:
        ValueError: wrong input poses or transl.

    Returns:
        Tuple[torch.Tensor]: Return converted poses, transl, pred_cam
            or verts, pred_cam.
    """
    R_, T_ = convert_world_view(R, T)

    RT = torch.eye(4, 4)[None]
    RT[:, :3, :3] = R_
    RT[:, :3, 3] = T_

    if verts is not None:
        poses = None
        betas = None
        transl = None
    else:
        assert poses is not None
        assert transl is not None
        if isinstance(poses, dict):
            poses = copy.deepcopy(poses)
            for k in poses:
                if isinstance(poses[k], np.ndarray):
                    poses[k] = torch.Tensor(poses[k])
        elif isinstance(poses, np.ndarray):
            poses = torch.Tensor(poses)
        elif isinstance(poses, torch.Tensor):
            poses = poses.clone()
        else:
            raise ValueError(f'Wrong data type of poses: {type(poses)}.')

        if isinstance(transl, np.ndarray):
            transl = torch.Tensor(transl)
        elif isinstance(transl, torch.Tensor):
            transl = transl.clone()
        else:
            raise ValueError('Should pass valid `transl`.')
        transl = transl.view(-1, 3)

        if isinstance(betas, np.ndarray):
            betas = torch.Tensor(betas)
        elif isinstance(betas, torch.Tensor):
            betas = betas.clone()

        body_model = get_body_model(
            body_model_dir, gender=gender, model_type=model_type)
        if isinstance(poses, dict):
            poses.update({'transl': transl, 'betas': betas})
        else:
            if isinstance(poses, np.ndarray):
                poses = torch.tensor(poses)
            poses = body_model.tensor2dict(
                full_pose=poses, transl=transl, betas=betas)
        verts = get_mesh_info(body_model, **poses)['vertices']

        global_orient = poses['global_orient']
        global_orient = rotmat_to_aa(R_ @ aa_to_rotmat(global_orient))
        poses['global_orient'] = global_orient
        poses['transl'] = None
        verts_rotated = get_mesh_info(body_model, **poses)['vertices']
        rotated_pose = body_model.dict2tensor(poses)

    verts_converted = verts.clone().view(-1, 3)
    verts_converted = RT @ torch.cat(
        [verts_converted,
         torch.ones(verts_converted.shape[0], 1)], dim=-1).unsqueeze(-1)
    verts_converted = verts_converted.squeeze(-1)
    verts_converted = verts_converted[:, :3] / verts_converted[:, 3:]
    verts_converted = verts_converted.view(verts.shape[0], -1, 3)
    num_frame = verts_converted.shape[0]
    if poses is not None:
        transl = torch.mean(verts_converted - verts_rotated, dim=1)

    pred_cam = None
    if K is not None:
        zmean = torch.mean(verts_converted, dim=1)[:, 2]

        K, _, _ = convert_cameras(
            K,
            is_perspective=True,
            convention_dst='opencv',
            convention_src='opencv',
            in_ndc_dst=True,
            in_ndc_src=False,
            resolution_src=resolution)
        K = K.repeat(num_frame, 1, 1)

        # transl_ = torch.zeros_like(ppoi
        print('transl_added')
        pred_cam = convert_perspective_to_weakperspective(
            K=K, zmean=zmean, in_ndc=True, resolution=resolution)
        if poses is not None:
            pred_cam[:, 2] += transl[:, 0]
            pred_cam[:, 3] += transl[:, 1]
    if poses is not None:
        return rotated_pose, pred_cam
    else:
        return verts_converted, pred_cam
