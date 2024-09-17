# Copyright (c) OpenMMLab. All rights reserved.

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.utils.geometry import rotation_matrix_to_angle_axis
from mmhuman3d.utils.transforms import aa_to_rotmat


class STAR(nn.Module):
    NUM_JOINTS = 24
    NUM_VERTS = 6890
    NUM_FACES = 13776

    def __init__(self,
                 model_path: str,
                 gender: str = 'neutral',
                 keypoint_src: str = 'star',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 create_global_orient: bool = True,
                 global_orient: Optional[torch.Tensor] = None,
                 create_body_pose: bool = True,
                 body_pose: torch.Tensor = None,
                 num_betas: int = 10,
                 create_betas: bool = True,
                 betas: torch.Tensor = None,
                 create_transl: bool = True,
                 transl: torch.Tensor = None,
                 batch_size: int = 1,
                 dtype: torch.dtype = torch.float32) -> None:
        """STAR model constructor.

        Args:
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored.
            gender: str, optional
                Which gender to load.
            keypoint_src: str
                Source convention of keypoints. This convention is used for
                keypoints obtained from joint regressors. Keypoints then
                undergo  conversion into keypoint_dst convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(3*23)
                The default value for the body pose variable.
                (default = None)
            num_betas: int, optional
                Number of shape components to use
                (default = 10).
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            batch_size: int, optional
                The batch size used for creating the member variables.
            dtype: torch.dtype, optional
                The data type for the created variables.
        """
        if gender not in ['male', 'female', 'neutral']:
            raise RuntimeError('Invalid gender! Should be one of '
                               '[\'male\', \'female\', or \'neutral\']!')
        self.gender = gender

        model_fname = 'STAR_{}.npz'.format(gender.upper())
        if not os.path.exists(model_path):
            raise RuntimeError('Path {} does not exist!'.format(model_path))
        elif os.path.isdir(model_path):
            star_path = os.path.join(model_path, model_fname)
        else:
            if os.path.split(model_path)[-1] != model_fname:
                raise RuntimeError(
                    f'Model filename ({model_fname}) and gender '
                    f'({gender}) are incompatible!')
            star_path = model_path

        super(STAR, self).__init__()

        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        star_model = np.load(star_path, allow_pickle=True)
        J_regressor = star_model['J_regressor']
        self.num_betas = num_betas

        # Model sparse joints regressor, regresses joints location from a mesh
        self.register_buffer('J_regressor',
                             torch.tensor(J_regressor, dtype=torch.float))

        # Model skinning weights
        self.register_buffer(
            'weights', torch.tensor(star_model['weights'], dtype=torch.float))

        # Model pose corrective blend shapes
        self.register_buffer(
            'posedirs',
            torch.tensor(
                star_model['posedirs'].reshape((-1, 93)), dtype=torch.float))

        # Mean Shape
        self.register_buffer(
            'v_template',
            torch.tensor(star_model['v_template'], dtype=torch.float))

        # Shape corrective blend shapes
        self.register_buffer(
            'shapedirs',
            torch.tensor(
                star_model['shapedirs'][:, :, :num_betas], dtype=torch.float))

        # Mesh traingles
        self.register_buffer(
            'faces', torch.from_numpy(star_model['f'].astype(np.int64)))
        self.f = star_model['f']
        self.register_buffer('faces_tensor',
                             torch.from_numpy(star_model['f'].astype(
                                 np.int64)))  # alias for face tensor in render

        # Kinematic tree of the model
        self.register_buffer(
            'kintree_table',
            torch.from_numpy(star_model['kintree_table'].astype(np.int64)))

        id_to_col = {
            self.kintree_table[1, i].item(): i
            for i in range(self.kintree_table.shape[1])
        }
        self.register_buffer(
            'parent',
            torch.LongTensor([
                id_to_col[self.kintree_table[0, it].item()]
                for it in range(1, self.kintree_table.shape[1])
            ]))

        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3],
                                                    dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(
                        global_orient, dtype=dtype)

            global_orient = nn.Parameter(
                default_global_orient, requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_JOINTS, 3, 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose, dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.num_betas],
                                            dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                'betas', nn.Parameter(default_betas, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl', nn.Parameter(default_transl, requires_grad=True))

        self.verts = None
        self.J = None
        self.R = None

    def forward(self,
                body_pose: Optional[torch.Tensor] = None,
                global_orient: Optional[torch.Tensor] = None,
                betas: Optional[torch.Tensor] = None,
                transl: Optional[torch.Tensor] = None,
                gender: Optional[str] = None,
                return_verts: bool = True,
                return_full_pose: bool = True,
                **kwargs) -> torch.Tensor:
        """Forward pass for the STAR model.

        Args:
            body_pose: torch.Tensor, shape Bx23x3x3 tensor.
                Pose parameters for the STAR model. It should be a tensor that
                contains joint rotations in axis-angle format. If given, ignore
                the member variable and use it as the body parameters.
                (default=None)
            global_orient: torch.Tensor, shape Bx1x3x3 tensor. (default=None)
            betas: torch.Tensor, shape Bx10
                Shape parameters for the STAR model. If given, ignore the
                member variable and use it as shape parameters. (default=None)
            transl: torch.Tensor, shape Bx3
                Translation vector for the STAR model. If given, ignore the
                member variable and use it as the translation of the body.
                (default=None)
        Returns:
            output: Contains output parameters and attributes corresponding
            to other body models.
        """
        body_pose = body_pose if body_pose is not None else self.body_pose
        device = body_pose.device

        if body_pose.shape[1] % 24 != 0:
            body_pose = torch.cat((global_orient, body_pose), dim=1)
        betas = betas if betas is not None else self.betas
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        batch_size = body_pose.shape[0]
        if body_pose.shape[1] == 72:
            body_pose = body_pose.view(batch_size, -1, 3)
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, self.num_betas)[None, :].expand(
            batch_size, -1, -1)
        beta = betas[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        if len(body_pose.shape) == 4:
            # the shape of body pose is rot matrix, convert to angle_axis
            body_pose = rotation_matrix_to_angle_axis(
                body_pose.view(-1, 3, 3)).view(batch_size, -1, 3)

        pose_quat = self.quat_feat(body_pose.view(-1, 3)).view(batch_size, -1)
        pose_feat = torch.cat((pose_quat[:, 4:], beta[:, 1]), 1)

        R = aa_to_rotmat(body_pose.view(-1, 3)).view(batch_size, 24, 3, 3)
        R = R.view(batch_size, 24, 3, 3)

        posedirs = self.posedirs[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(
            posedirs, pose_feat[:, :, None]).view(-1, 6890, 3)

        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0,
                                     1]).to(device).view(1, 1, 1, 4).expand(
                                         batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)],
                         dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights,
                         G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(
                             6890, batch_size, 4, 4).transpose(0, 1)
        rest_shape_h = torch.cat(
            [v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        v = v + transl[:, None, :]
        v.f = self.f
        v.v_posed = v_posed
        v.v_shaped = v_shaped

        root_transform = self.with_zeros(
            torch.cat((R[:, 0], J[:, 0][:, :, None]), 2))
        results = [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i = self.with_zeros(
                torch.cat((R[:, i + 1], J[:, i + 1][:, :, None] -
                           J[:, self.parent[i]][:, :, None]), 2))
            curr_res = torch.matmul(results[self.parent[i]], transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        v.J_transformed = posed_joints + transl[:, None, :]

        joints, joint_mask = convert_kps(
            v.J_transformed,
            src=self.keypoint_src,
            dst=self.keypoint_dst,
            approximate=self.keypoint_approximate)

        joint_mask = torch.tensor(
            joint_mask, dtype=torch.uint8, device=joints.device)
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)

        global_orient = body_pose[:, 0, :][:, None]
        body_pose = body_pose[:, 1:, :]

        output = dict(
            global_orient=global_orient,
            body_pose=body_pose,
            joints=posed_joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=beta)

        if return_verts:
            output['vertices'] = v
        if return_full_pose:
            output['full_pose'] = torch.cat([global_orient, body_pose], dim=1)

        return output

    @classmethod
    def with_zeros(self, input):
        """Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor.

        :param input: A tensor of dimensions batch size x 3 x 4
        :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
        """
        batch_size = input.shape[0]
        row_append = torch.FloatTensor(([0.0, 0.0, 0.0, 1.0])).to(input.device)
        row_append.requires_grad = False
        padded_tensor = torch.cat(
            [input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)
        return padded_tensor

    @classmethod
    def quat_feat(self, theta):
        """Computes a normalized quaternion ([0,0,0,0]  when the body is in
        rest pose) given joint angles.

        :param theta: A tensor of joints axis angles.
        :return:
        """
        l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_sin * normalized, v_cos - 1], dim=1)
        return quat

    def name(self) -> str:
        return 'STAR'
