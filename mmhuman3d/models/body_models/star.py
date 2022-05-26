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

import os

import numpy as np
import torch
import torch.nn as nn


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
