import copy
import os
from typing import Iterable, Optional, Union

import numpy as np
import torch
from pytorch3d.renderer.cameras import CamerasBase

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.cameras import (
    convert_cameras,
    convert_perspective_to_weakperspective,
    convert_world_view,
)
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa


def convert_smpl_from_opencv_calibration(
        R: Union[np.ndarray, torch.Tensor],
        T: Union[np.ndarray, torch.Tensor],
        K: Optional[Union[np.ndarray, torch.Tensor]] = None,
        resolution: Optional[Union[Iterable[int], int]] = None,
        verts: Optional[Union[np.ndarray, torch.Tensor]] = None,
        poses: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
        model_path: Optional[str] = None,
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
        model_path (Optional[str], optional): model path.
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

        body_model = build_body_model(
            dict(
                type=model_type,
                model_path=os.path.join(model_path, model_type),
                gender=gender,
                model_type=model_type))
        if isinstance(poses, dict):
            poses.update({'transl': transl, 'betas': betas})
        else:
            if isinstance(poses, np.ndarray):
                poses = torch.tensor(poses)
            poses = body_model.tensor2dict(
                full_pose=poses, transl=transl, betas=betas)
        model_output = body_model(**poses)
        verts = model_output['vertices']

        global_orient = poses['global_orient']
        global_orient = rotmat_to_aa(R_ @ aa_to_rotmat(global_orient))
        poses['global_orient'] = global_orient
        poses['transl'] = None
        verts_rotated = model_output['vertices']
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

    orig_cam = None
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

        orig_cam = convert_perspective_to_weakperspective(
            K=K, zmean=zmean, in_ndc=True, resolution=resolution)
        if poses is not None:
            orig_cam[:, 2] += transl[:, 0]
            orig_cam[:, 3] += transl[:, 1]
    if poses is not None:
        return rotated_pose, orig_cam
    else:
        return verts_converted, orig_cam


def project_points(points3d: Union[np.ndarray, torch.Tensor],
                   cameras: CamerasBase = None,
                   resolution: Iterable[int] = None,
                   K: Union[torch.Tensor, np.ndarray] = None,
                   R: Union[torch.Tensor, np.ndarray] = None,
                   T: Union[torch.Tensor, np.ndarray] = None,
                   convention: str = 'opencv',
                   in_ndc: bool = False) -> Union[torch.Tensor, np.ndarray]:
    """Project 3d points to image.

    Args:
        points3d (Union[np.ndarray, torch.Tensor]): shape could be (..., 3).
        cameras (CamerasBase): pytorch3d cameras or mmhuman3d cameras.
        resolution (Iterable[int]): (height, width) for rectangle or width for
            square.
        K (Union[torch.Tensor, np.ndarray], optional): intrinsic matrix.
            Defaults to None.
        R (Union[torch.Tensor, np.ndarray], optional): rotation matrix.
            Defaults to None.
        T (Union[torch.Tensor, np.ndarray], optional): translation matrix.
            Defaults to None.
        convention (str, optional): camera convention. Defaults to 'opencv'.
        in_ndc (bool, optional): whether in NDC. Defaults to False.

    Returns:
        Union[torch.Tensor, np.ndarray]: transformed points of shape (..., 2).
    """
    if cameras is None:
        cameras = build_cameras(
            dict(
                type='perspective',
                convention=convention,
                in_ndc=in_ndc,
                resolution=resolution,
                K=K,
                R=R,
                T=T))
    if cameras.get_image_size() is not None:
        image_size = cameras.get_image_size()
    else:
        image_size = resolution
    if isinstance(points3d, np.ndarray):
        points3d = torch.Tensor(points3d[..., :3]).to(cameras.device)
        points2d = cameras.transform_points_screen(
            points3d, image_size=image_size).cpu().numpy()
    elif isinstance(points3d, torch.Tensor):
        points3d = points3d[..., :3].to(cameras.device)
        points2d = cameras.transform_points_screen(
            points3d, image_size=image_size)
    return points2d
