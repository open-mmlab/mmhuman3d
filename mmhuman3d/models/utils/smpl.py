# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from smplx.lbs import vertices2joints

from ..builder import BODY_MODELS

try:
    from smplx import SMPL as SMPL_
    has_smpl = True
except (ImportError, ModuleNotFoundError):
    has_smpl = False


@BODY_MODELS.register_module()
class SMPL(nn.Module):
    """SMPL 3d human mesh model of paper ref: Matthew Loper.

    ``SMPL: A skinned
    multi-person linear model''. This module is based on the smplx project
    (https://github.com/vchoutas/smplx).
    Args:
        smpl_path (str): The path to the folder where the model weights are
            stored.
        joints_regressor (str): The path to the file where the joints
            regressor weight are stored.
    """

    def __init__(self,
                 smpl_path,
                 joints_regressor,
                 extra_joints_regressor=None):
        super().__init__()

        assert has_smpl, 'Please install smplx to use SMPL.'

        self.smpl_neutral = SMPL_(
            model_path=smpl_path,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            gender='neutral')

        self.smpl_male = SMPL_(
            model_path=smpl_path,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            gender='male')

        self.smpl_female = SMPL_(
            model_path=smpl_path,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            gender='female')

        joints_regressor = torch.tensor(
            np.load(joints_regressor), dtype=torch.float)[None, ...]
        self.register_buffer('joints_regressor', joints_regressor)
        if extra_joints_regressor is not None:
            extra_joints_regressor = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
        self.register_buffer('extra_joints_regressor', extra_joints_regressor)

        self.num_verts = self.smpl_neutral.get_num_verts()
        self.num_joints = self.joints_regressor.shape[1]

    def smpl_forward(self, model, num_joints, **kwargs):
        """Apply a specific SMPL model with given model parameters.
        Note:
            B: batch size
            V: number of vertices
            K: number of joints
        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d joints regressed
                    from mesh vertices.
        """

        betas = kwargs['betas']
        batch_size = betas.shape[0]
        device = betas.device
        output = {}
        if batch_size == 0:
            output['vertices'] = betas.new_zeros([0, self.num_verts, 3])
            output['joints'] = betas.new_zeros([0, num_joints, 3])
        else:
            smpl_out = model(**kwargs)
            output['vertices'] = smpl_out.vertices
            if num_joints == 24:
                output['joints'] = torch.matmul(
                    self.joints_regressor.to(device), output['vertices'])
            else:
                joints_mapper = [
                    24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25,
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 8, 5, 45, 46, 4, 7, 21,
                    19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25,
                    28, 27
                ]
                extra_joints = vertices2joints(self.extra_joints_regressor,
                                               smpl_out.vertices)
                joints = torch.cat([smpl_out.joints, extra_joints], dim=1)
                joints = joints[:, joints_mapper, :]
                output['joints'] = joints

        return output

    def get_faces(self):
        """Return mesh faces.

        Note:
            F: number of faces
        Returns:
            faces: np.ndarray([F, 3]), mesh faces
        """
        return self.smpl_neutral.faces

    def forward(self,
                betas,
                body_pose,
                global_orient,
                num_joints=24,
                pose2rot=True,
                transl=None,
                gender=None):
        """Forward function.
        Note:
            B: batch size
            J: number of controllable joints of model, for smpl model J=23
            K: number of joints
        Args:
            betas: Tensor([B, 10]), human body shape parameters of SMPL model.
            body_pose: Tensor([B, J*3] or [B, J, 3, 3]), human body pose
                parameters of SMPL model. It should be axis-angle vector
                ([B, J*3]) or rotation matrix ([B, J, 3, 3)].
            global_orient: Tensor([B, 3] or [B, 1, 3, 3]), global orientation
                of human body. It should be axis-angle vector ([B, 3]) or
                rotation matrix ([B, 1, 3, 3)].
            transl: Tensor([B, 3]), global translation of human body.
            gender: Tensor([B]), gender parameters of human body. -1 for
                neutral, 0 for male , 1 for female.
        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d joints regressed from
                    mesh vertices.
        """

        batch_size = betas.shape[0]
        if batch_size > 0 and gender is not None:
            output = {
                'vertices': betas.new_zeros([batch_size, self.num_verts, 3]),
                'joints': betas.new_zeros([batch_size, self.num_joints, 3])
            }

            mask = gender < 0
            _out = self.smpl_forward(
                self.smpl_neutral,
                betas=betas[mask],
                body_pose=body_pose[mask],
                global_orient=global_orient[mask],
                transl=transl[mask] if transl is not None else None,
                pose2rot=pose2rot,
                num_joints=num_joints)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']

            mask = gender == 0
            _out = self.smpl_forward(
                self.smpl_male,
                betas=betas[mask],
                body_pose=body_pose[mask],
                global_orient=global_orient[mask],
                transl=transl[mask] if transl is not None else None,
                pose2rot=pose2rot,
                num_joints=num_joints)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']

            mask = gender == 1
            _out = self.smpl_forward(
                self.smpl_male,
                betas=betas[mask],
                body_pose=body_pose[mask],
                global_orient=global_orient[mask],
                transl=transl[mask] if transl is not None else None,
                pose2rot=pose2rot,
                num_joints=num_joints)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']
        else:
            return self.smpl_forward(
                self.smpl_neutral,
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                pose2rot=pose2rot,
                num_joints=num_joints)

        return output
