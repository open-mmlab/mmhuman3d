# Copyright (c) OpenMMLab. All rights reserved.

# Adapted from:
#
# https://github.com/ahmedosman/STAR/blob/master/star/pytorch/star.py
#
#
# -*- coding: utf-8 -*-
#
# LICENSE:
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import division
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .. import BODY_MODELS
from .utils import quat_feat, rodrigues, with_zeros


@BODY_MODELS.register_module(name=['STAR', 'star'])
class STAR(nn.Module):

    def __init__(self,
                 model_path: str,
                 gender: str = 'neutral',
                 num_betas: int = 10) -> None:
        """STAR model constructor.
        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        num_betas: int, optional
            Number of shape components to use
            (default = 10).
        gender: str, optional
            Which gender to load
        """
        if gender not in ['male', 'female', 'neutral']:
            raise RuntimeError('Invalid gender! Should be one of '
                               '[\'male\', \'female\', or \'neutral\']!')
        self.gender = gender

        if os.path.isdir(model_path):
            star_path = os.path.join(model_path,
                                     'STAR_{}.npz'.format(gender.upper()))
        else:
            star_path = model_path
        if not os.path.exists(star_path):
            raise RuntimeError('Path {} does not exist!'.format(star_path))

        super(STAR, self).__init__()

        star_model = np.load(star_path, allow_pickle=True)
        J_regressor = star_model['J_regressor']
        self.num_betas = num_betas

        # Model sparse joints regressor, regresses joints location from a mesh
        self.register_buffer('J_regressor',
                             torch.cuda.FloatTensor(J_regressor))

        # Model skinning weights
        self.register_buffer('weights',
                             torch.cuda.FloatTensor(star_model['weights']))
        # Model pose corrective blend shapes
        self.register_buffer(
            'posedirs',
            torch.cuda.FloatTensor(star_model['posedirs'].reshape((-1, 93))))
        # Mean Shape
        self.register_buffer('v_template',
                             torch.cuda.FloatTensor(star_model['v_template']))
        # Shape corrective blend shapes
        self.register_buffer(
            'shapedirs',
            torch.cuda.FloatTensor(
                np.array(star_model['shapedirs'][:, :, :num_betas])))
        # Mesh traingles
        self.register_buffer(
            'faces', torch.from_numpy(star_model['f'].astype(np.int64)))
        self.f = star_model['f']
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

        self.verts = None
        self.J = None
        self.R = None

    def forward(self,
                pose: Optional[torch.Tensor] = None,
                betas: Optional[torch.Tensor] = None,
                trans: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for the SMPL model.
        Parameters
        ----------
        :param pose: pose  parameters - A batch size x 72 tensor
            (3 numbers for each joint)
        :param beta: beta  parameters - A batch size x number of betas
        :param beta: trans parameters - A batch size x 3
        :return:
             v         : batch size x 6890 x 3
                         The STAR model vertices
             v.v_vposed: batch size x 6890 x 3 model
                         STAR vertices in T-pose after adding the shape
                         blend shapes and pose blend shapes
             v.v_shaped: batch size x 6890 x 3
                         STAR vertices in T-pose after adding the shape
                         blend shapes and pose blend shapes
             v.J_transformed:batch size x 24 x 3
                            Posed model joints.
             v.f: A numpy array of the model face.
        """
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, self.num_betas)[None, :].expand(
            batch_size, -1, -1)
        beta = betas[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        pose_quat = quat_feat(pose.view(-1, 3)).view(batch_size, -1)
        pose_feat = torch.cat((pose_quat[:, 4:], beta[:, 1]), 1)

        R = rodrigues(pose.view(-1, 3)).view(batch_size, 24, 3, 3)
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
        v = v + trans[:, None, :]
        v.f = self.f
        v.v_posed = v_posed
        v.v_shaped = v_shaped

        root_transform = with_zeros(
            torch.cat((R[:, 0], J[:, 0][:, :, None]), 2))
        results = [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i = with_zeros(
                torch.cat((R[:, i + 1], J[:, i + 1][:, :, None] -
                           J[:, self.parent[i]][:, :, None]), 2))
            curr_res = torch.matmul(results[self.parent[i]], transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        v.J_transformed = posed_joints + trans[:, None, :]
        return v
